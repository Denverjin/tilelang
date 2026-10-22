"""Explicit layout aliases share the physical clock of retimed L0B owners."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_ascend_major_k_layout, make_ascend_major_mn_layout
from tvm import tirx
from testing.ascend._ir import allocated_buffer, nodes


def _retimed_layout_pair(qk_steps=4, pv_steps=4, alias_kind=None, transpose_pv=False, annotated=True):
    @T.prim_func
    def main(
        Q: T.Tensor((64, 128), "bfloat16"),
        K: T.Tensor((64, 128), "bfloat16"),
        CQ: T.Tensor((3, 64, 64), "float32"),
        CV: T.Tensor((3, pv_steps, 64, 128), "float32"),
    ):
        with T.Kernel(1):
            q = T.alloc_l1((64, 128), "bfloat16")
            k0 = T.alloc_l1((32, 128), "bfloat16")
            k1 = T.alloc_l1((32, 128), "bfloat16")
            aq = T.alloc_l0a((64, 128), "bfloat16")
            bq = T.alloc_l0b((64, 128), "bfloat16")
            if alias_kind == "view":
                bv = T.view(bq, (128, 64) if transpose_pv else (64, 128))
            elif alias_kind == "reshape":
                bv = T.reshape(bq, (128, 64) if transpose_pv else (64, 128))
            else:
                bv = T.alloc_l0b((64, 128), "bfloat16")
            cq = T.alloc_l0c((64, 64), "float32")
            cv = T.alloc_l0c((pv_steps, 64, 128), "float32")
            T.annotate_buffer_versions({q: 1, k0: 1, k1: 1, aq: 4, bq: 4, bv: 4, cq: 1, cv: 1})
            if annotated:
                T.annotate_layout(
                    {
                        bq: make_ascend_major_k_layout(bq),
                        bv: make_ascend_major_k_layout(bv) if transpose_pv else make_ascend_major_mn_layout(bv),
                    }
                )
            T.copy(Q, q)
            T.copy(K[:32, :], k0)
            T.copy(K[32:, :], k1)
            # Keep the Cube mode fixed so this regression isolates storage
            # ownership from separately scheduled configuration changes.
            for step in T.Pipelined(3, num_stages=2, annotations={"enable_offset": True}):
                if alias_kind is None:
                    T.assume_conflict(bq, bv)
                with T.Stage(0):
                    for d in T.Pipelined(qk_steps, num_stages=4):
                        with T.Stage(0):
                            T.copy(q, aq)
                            with T.Task():
                                T.copy(k0, bq[:32, :], l0_pitch=4)
                                T.copy(k1, bq[32:, :], l0_pitch=4)
                            T.gemm(aq, bq, cq, transpose_B=True, clear_accum=d == 0)
                    T.copy(cq, CQ[step, :, :])
                with T.Stage(1):
                    for _region in T.Pipelined(1, num_stages=4):
                        with T.Stage(0):
                            T.copy(q[:, :64], aq[:, :64])
                        for d in T.Pipelined(pv_steps, num_stages=2):
                            with T.Stage(0):
                                with T.Task():
                                    if transpose_pv:
                                        T.copy(k0, bv[:, :32], transpose=True, l0_pitch=8)
                                        T.copy(k1, bv[:, 32:], transpose=True, l0_pitch=8)
                                    else:
                                        T.copy(k0, bv[:32, :], l0_pitch=8)
                                        T.copy(k1, bv[32:, :], l0_pitch=8)
                                T.gemm(aq[:, :64], bv, cv[d, :, :], transpose_B=transpose_pv, clear_accum=True)
                        for d in T.Pipelined(pv_steps, num_stages=2):
                            with T.Stage(0):
                                T.copy(cv[d, :, :], CV[step, d, :, :])

    return main


@pytest.mark.parametrize("qk_steps,pv_steps", [(4, 4), (3, 2)])
@pytest.mark.parametrize("disable_reuse", [False, True])
def test_default_conflict_hint_shares_retimed_owner_clock(qk_steps, pv_steps, disable_reuse):
    snapshots = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.PrepareMultiBuffer", "tl.MergeUBAllocations"):
                snapshots[info.name] = mod

    with (
        tvm.target.Target("ascend"),
        tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}, instruments=[Capture()]),
    ):
        tilelang.lower(_retimed_layout_pair(qk_steps, pv_steps), target="ascend", enable_device_compile=False)

    prepared = snapshots["tl.PrepareMultiBuffer"]
    bq, bv = [allocated_buffer(prepared, name) for name in ("bq", "bv")]
    assert not bq.data.same_as(bv.data)
    clocks = []
    for loop in nodes(prepared, tirx.For):
        if "tl.multi_buffer_counter_map" not in loop.annotations:
            continue
        counters = [
            counter
            for storage, counter in loop.annotations["tl.multi_buffer_counter_map"].items()
            if storage.same_as(bq.data) or storage.same_as(bv.data)
        ]
        if not counters:
            continue
        assert len(counters) == 2
        assert counters[0].same_as(counters[1])
        clocks.append(counters[0])
    assert len(clocks) == 2
    assert all(clock.same_as(clocks[0]) for clock in clocks)
    advances = [
        store
        for store in nodes(prepared, tirx.BufferStore)
        if store.buffer.same_as(clocks[0]) and tvm.arith.Analyzer().can_prove_equal(store.value, clocks[0][0] + 1)
    ]
    assert len(advances) == 2

    allocations = [
        node.buffer
        for func in snapshots["tl.MergeUBAllocations"].functions.values()
        for node in nodes(func, tirx.AllocBuffer)
        if node.buffer.scope() in ("shared.l0b", "shared.l0b.dyn")
    ]
    assert len(allocations) == 1
    allocated_bytes = int(allocations[0].shape[0]) * tvm.DataType(allocations[0].dtype).bits // 8
    assert allocated_bytes == (131072 if disable_reuse else 65536)


@pytest.mark.parametrize("alias_kind", ["view", "reshape"])
@pytest.mark.parametrize("transpose_pv,annotated", [(False, True), (True, True), (False, False), (True, False)])
def test_storage_views_keep_layouts_and_share_one_version_clock(alias_kind, transpose_pv, annotated):
    snapshots = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in (
                "tl.NormalizeAscendFractalStorage",
                "tl.PrepareMultiBuffer",
                "tl.MaterializeMultiBuffer",
                "tl.MergeUBAllocations",
            ):
                snapshots[info.name] = mod

    with tvm.target.Target("ascend"), tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}, instruments=[Capture()]):
        tilelang.lower(_retimed_layout_pair(3, 2, alias_kind, transpose_pv, annotated), target="ascend", enable_device_compile=False)

    def l0b_layouts(mod):
        return {
            buffer: layout
            for block in nodes(mod, tirx.SBlock)
            for buffer, layout in block.annotations.get("layout_map", {}).items()
            if buffer.scope() == "shared.l0b"
        }

    layouts = l0b_layouts(snapshots["tl.NormalizeAscendFractalStorage"])
    assert len(layouts) == 2
    first, second = layouts
    assert first.data.same_as(second.data)
    assert not layouts[first].is_equal(layouts[second])
    expected_shapes = {(64, 128), (128, 64)} if transpose_pv else {(64, 128)}
    assert {tuple(int(x) for x in buffer.shape) for buffer in layouts} == expected_shapes

    clocks = [
        counter
        for loop in nodes(snapshots["tl.PrepareMultiBuffer"], tirx.For)
        for storage, counter in loop.annotations.get("tl.multi_buffer_counter_map", {}).items()
        if storage.same_as(first.data)
    ]
    assert len(clocks) == 2 and clocks[0].same_as(clocks[1])
    expanded = {buffer: layout for buffer, layout in l0b_layouts(snapshots["tl.MaterializeMultiBuffer"]).items() if len(buffer.shape) == 3}
    assert len(expanded) == 2
    for buffer, layout in expanded.items():
        assert int(buffer.shape[0]) == 4 and int(buffer.strides[0]) == 8192
        expected = make_ascend_major_k_layout(buffer) if transpose_pv or buffer.name == "bq" else make_ascend_major_mn_layout(buffer)
        assert layout.is_equal(expected)
    allocations = [
        node.buffer
        for func in snapshots["tl.MergeUBAllocations"].functions.values()
        for node in nodes(func, tirx.AllocBuffer)
        if node.buffer.scope().startswith("shared.l0b")
    ]
    assert len(allocations) == 1
    assert int(allocations[0].shape[0]) * tvm.DataType(allocations[0].dtype).bits // 8 == 65536


def test_conflict_hint_does_not_join_nested_storage_owners():
    @T.prim_func
    def main(Q: T.Tensor((64, 128), "bfloat16"), K: T.Tensor((64, 128), "bfloat16"), C: T.Tensor((64, 128), "float32")):
        with T.Kernel(1):
            q = T.alloc_l1((64, 128), "bfloat16")
            k = T.alloc_l1((64, 128), "bfloat16")
            aq = T.alloc_l0a((64, 128), "bfloat16")
            av = T.alloc_l0a((64, 64), "bfloat16")
            bq = T.alloc_l0b((64, 128), "bfloat16")
            bv = T.alloc_l0b((64, 128), "bfloat16")
            cq = T.alloc_l0c((64, 64), "float32")
            cv = T.alloc_l0c((64, 128), "float32")
            T.annotate_buffer_versions({q: 1, k: 1, aq: 1, av: 1, bq: 4, bv: 4, cq: 1, cv: 1})
            T.copy(Q, q)
            T.copy(K, k)
            for _i in T.Pipelined(2, num_stages=2):
                T.assume_conflict(bq, bv)
                with T.Stage(0):
                    T.copy(q, aq)
                    T.copy(k, bq)
                    T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)
                    T.copy(cq, C[:, :64])
                    for _j in T.Pipelined(2, num_stages=2):
                        with T.Stage(0):
                            T.copy(q[:, :64], av)
                            T.copy(k, bv)
                            T.gemm(av, bv, cv, clear_accum=True)
                            T.copy(cv, C)

    with tvm.target.Target("ascend"):
        tilelang.lower(main, target="ascend", enable_device_compile=False)


@pytest.mark.parametrize(
    "automatic,staged_second",
    [(False, False), (False, True), (True, False)],
    ids=["manual-first", "manual-second", "automatic-offset"],
)
@pytest.mark.parametrize("disable_reuse", [False, True])
def test_conflict_hint_keeps_independent_owners_when_stages_can_differ(automatic, staged_second, disable_reuse):
    @T.prim_func
    def main(Q: T.Tensor((16, 16), "bfloat16"), K: T.Tensor((16, 16), "bfloat16"), C: T.Tensor((2, 16, 16), "float32")):
        with T.Kernel(1):
            q = T.alloc_l1((16, 16), "bfloat16")
            k = T.alloc_l1((16, 16), "bfloat16")
            aq = T.alloc_l0a((16, 16), "bfloat16")
            av = T.alloc_l0a((16, 16), "bfloat16")
            bq = T.alloc_l0b((16, 16), "bfloat16")
            bv = T.alloc_l0b((16, 16), "bfloat16")
            cq = T.alloc_l0c((16, 16), "float32")
            cv = T.alloc_l0c((16, 16), "float32")
            T.annotate_buffer_versions({q: 1, k: 1, aq: 4, av: 1, bq: 4, bv: 4, cq: 1, cv: 1})
            T.assume_conflict(bq, bv)
            T.copy(Q, q)
            T.copy(K, k)
            T.copy(q, av)
            staged = bv if staged_second else bq
            other = bq if staged_second else bv
            for _i in T.Pipelined(4, num_stages=4, annotations={"enable_offset": True}):
                if automatic:
                    T.copy(q, aq)
                    T.copy(k, staged)
                    T.gemm(aq, staged, cq, transpose_B=True, clear_accum=True)
                else:
                    with T.Stage(0):
                        T.copy(q, aq)
                        T.copy(k, staged)
                    with T.Stage(1):
                        T.gemm(aq, staged, cq, transpose_B=True, clear_accum=True)
            T.copy(cq, C[0, :, :])
            for _j in T.Pipelined(4, num_stages=4):
                with T.Stage(0):
                    T.copy(k, other)
                    T.gemm(av, other, cv, transpose_B=True, clear_accum=True)
            T.copy(cv, C[1, :, :])

    snapshots = {}

    @tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.PrepareMultiBuffer", "tl.MaterializeMultiBuffer"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}, instruments=[Capture()]):
        artifact = tilelang.lower(main, target="ascend", enable_device_compile=False)

    prepared = snapshots["tl.PrepareMultiBuffer"]
    owners = []
    for name in ("staged", "other"):
        buffer = allocated_buffer(prepared, name)
        (owner,) = [loop for loop in nodes(prepared, tirx.For) if buffer.data in loop.annotations.get("multi_buffer_eligible", [])]
        owners.append(owner)
        assert int(allocated_buffer(snapshots["tl.MaterializeMultiBuffer"], name).shape[0]) == 4
    assert not owners[0].same_as(owners[1])
    assert artifact.kernel_source


@pytest.mark.parametrize(
    "alias_kind,transpose_pv,annotated", [(None, False, True), ("view", False, True), ("reshape", True, True), ("view", True, False)]
)
@tilelang.testing.requires_ascend
def test_retimed_layout_pair_values_across_owner_lengths_and_launches(alias_kind, transpose_pv, annotated):
    import torch

    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    with tvm.target.Target("ascend"):
        kernel = tilelang.compile(_retimed_layout_pair(3, 2, alias_kind, transpose_pv, annotated), target="ascend", out_idx=[2, 3])
    for seed in (7, 11, 19):
        torch.manual_seed(seed)
        q = torch.randint(-2, 3, (64, 128), dtype=torch.int32).to(dtype=torch.bfloat16, device="npu")
        k = torch.randint(-2, 3, (64, 128), dtype=torch.int32).to(dtype=torch.bfloat16, device="npu")
        cq, cv = kernel(q, k)
        torch.npu.synchronize()
        expected_qk = 3 * (q.cpu().float() @ k.cpu().float().T)
        expected_pv = q.cpu().float()[:, :64] @ k.cpu().float()
        torch.testing.assert_close(cq.cpu(), expected_qk.expand(3, 64, 64), rtol=0, atol=0)
        torch.testing.assert_close(cv.cpu(), expected_pv.expand(3, 2, 64, 128), rtol=0, atol=0)
