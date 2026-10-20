"""The reuse contract survives scheduled lowering and is consumed by packing."""

import pytest
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import calls, nodes
from testing.ascend.auto_schedule._reuse_utils import _make_cyclic_scalar_program


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_conflict_hint_reuses_independent_l0_layouts(disable_reuse):
    @T.prim_func
    def main(
        Q: T.Tensor((64, 128), "bfloat16"),
        K: T.Tensor((64, 128), "bfloat16"),
        CQ: T.Tensor((64, 64), "float32"),
        CV: T.Tensor((64, 128), "float32"),
    ):
        with T.Kernel(1):
            q = T.alloc_l1((64, 128), "bfloat16")
            k0 = T.alloc_l1((32, 128), "bfloat16")
            k1 = T.alloc_l1((32, 128), "bfloat16")
            aq = T.alloc_l0a((64, 128), "bfloat16")
            av = T.alloc_l0a((64, 64), "bfloat16")
            bq = T.alloc_l0b((64, 128), "bfloat16")
            bv = T.alloc_l0b((64, 128), "bfloat16")
            cq = T.alloc_l0c((64, 64), "float32")
            cv = T.alloc_l0c((64, 128), "float32")
            T.annotate_buffer_versions({q: 1, k0: 1, k1: 1, aq: 2, av: 1, bq: 4, bv: 4, cq: 1, cv: 1})
            T.assume_conflict(bq, bv, level=-1, cross=False)
            T.copy(Q, q)
            T.copy(K[:32, :], k0)
            T.copy(K[32:, :], k1)
            T.set_mmad_direction("m")
            for _d in T.Pipelined(4, num_stages=4):
                with T.Stage(0):
                    T.copy(q, aq)
                    with T.Task():
                        T.copy(k0, bq[:32, :], l0_pitch=4)
                        T.copy(k1, bq[32:, :], l0_pitch=4)
                    T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)
            T.copy(cq, CQ)
            T.copy(q[:, :64], av)
            T.set_mmad_direction("n")
            for _d in T.Pipelined(4, num_stages=4):
                with T.Stage(0):
                    with T.Task():
                        T.copy(k0, bv[:32, :], l0_pitch=8)
                        T.copy(k1, bv[32:, :], l0_pitch=8)
                    T.gemm(av, bv, cv, clear_accum=True)
            T.copy(cv, CV)

    snapshots = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AscendLowerTileOp", "tl.MergeUBAllocations"):
                snapshots[info.name] = mod

    with (
        tvm.target.Target("ascend"),
        tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}, instruments=[Capture()]),
    ):
        tilelang.lower(main, target="ascend", enable_device_compile=False)
    (allocation,) = [
        node.buffer
        for func in snapshots["tl.MergeUBAllocations"].functions.values()
        for node in nodes(func, tirx.AllocBuffer)
        if node.buffer.scope() in ("shared.l0b", "shared.l0b.dyn")
    ]
    allocated_bytes = int(allocation.shape[0]) * tvm.DataType(allocation.dtype).bits // 8
    assert allocated_bytes == (131072 if disable_reuse else 65536)

    offsets = {0: [], 1: []}
    analyzer = tvm.arith.Analyzer()
    for call in calls(snapshots["tl.AscendLowerTileOp"], "tl.ascend_load_cbuf_to_cb"):
        pointer = call.args[0]
        assert pointer.op.name == "tirx.tvm_access_ptr"
        offsets[int(call.args[-1])].append(int(analyzer.simplify(pointer.args[2] % 8192)))
    assert sorted(offsets[0]) == [0, 512]
    assert sorted(offsets[1]) == [0, 4096]


def test_alias_contract_lifetime():
    snapshots = {}
    wanted = {
        "tl.InsertSync",
        "tl.LowerScheduledTIR",
        "tl.MergeUBAllocations",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(_make_cyclic_scalar_program(), target="ascend")

    assert any("tl.buffer_alias_map" in block.annotations for block in nodes(snapshots["tl.InsertSync"], tirx.SBlock))
    assert any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.LowerScheduledTIR"], tirx.AttrStmt))
    assert not any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.MergeUBAllocations"], tirx.AttrStmt))


def test_fixed_versions_allow_ub_reuse_above_raw_capacity():
    @T.prim_func
    def main(x: T.Tensor((8,), "int32"), out: T.Tensor((8, 2), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((32768,), "int32")
            b = T.alloc_shared((32768,), "int32")
            T.annotate_buffer_versions({a: 1, b: 1})
            for i in T.Pipelined(8, num_stages=1):
                with T.Stage(0):
                    a[i * 4096] = x[i]
                with T.Stage(0):
                    out[i, 0] = a[i * 4096]
                with T.Stage(0):
                    b[i * 4096] = x[i] + 1
                with T.Stage(0):
                    out[i, 1] = b[i * 4096]

    allocations = []

    @tvm.ir.instrument.pass_instrument
    class CaptureAllocations:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                allocations.extend(
                    node.buffer
                    for func in mod.functions.values()
                    for node in nodes(func, tirx.AllocBuffer)
                    if node.buffer.scope() == "shared.dyn"
                )

    with tvm.transform.PassContext(instruments=[CaptureAllocations()]):
        tilelang.lower(main, target="ascend", enable_device_compile=False)

    # The raw sum is 256 KiB, but disjoint lifetimes need only 128 KiB of UB.
    (allocation,) = allocations
    assert allocation.dtype == "uint8"
    assert tuple(int(dim) for dim in allocation.shape) == (131072,)
