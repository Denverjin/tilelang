"""Explicit layout aliases share the physical clock of retimed L0B owners."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_ascend_major_k_layout, make_ascend_major_mn_layout
from tvm import tirx
from testing.ascend._ir import allocated_buffer, nodes


def _retimed_layout_pair(qk_steps=4, pv_steps=4):
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
            bv = T.alloc_l0b((64, 128), "bfloat16")
            cq = T.alloc_l0c((64, 64), "float32")
            cv = T.alloc_l0c((pv_steps, 64, 128), "float32")
            T.annotate_buffer_versions({q: 1, k0: 1, k1: 1, aq: 4, bq: 4, bv: 4, cq: 1, cv: 1})
            T.annotate_layout({bq: make_ascend_major_k_layout(bq), bv: make_ascend_major_mn_layout(bv)})
            T.copy(Q, q)
            T.copy(K[:32, :], k0)
            T.copy(K[32:, :], k1)
            for step in T.Pipelined(3, num_stages=2, annotations={"enable_offset": True}):
                T.assume_conflict(bq, bv)
                with T.Stage(0):
                    T.set_mmad_direction("m")
                    for d in T.Pipelined(qk_steps, num_stages=4):
                        with T.Stage(0):
                            T.copy(q, aq)
                            with T.Task():
                                T.copy(k0, bq[:32, :], l0_pitch=4)
                                T.copy(k1, bq[32:, :], l0_pitch=4)
                            T.gemm(aq, bq, cq, transpose_B=True, clear_accum=d == 0)
                    T.copy(cq, CQ[step, :, :])
                with T.Stage(1), T.IssuePhase({"MTE1": 1, "M": 1}):
                    T.set_mmad_direction("n")
                    for _region in T.Pipelined(1, num_stages=4):
                        with T.Stage(0):
                            T.copy(q[:, :64], aq[:, :64])
                        for d in T.Pipelined(pv_steps, num_stages=2):
                            with T.Stage(0):
                                with T.Task():
                                    T.copy(k0, bv[:32, :], l0_pitch=8)
                                    T.copy(k1, bv[32:, :], l0_pitch=8)
                                T.gemm(aq[:, :64], bv, cv[d, :, :], clear_accum=True)
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


@tilelang.testing.requires_ascend
def test_retimed_layout_pair_values_across_owner_lengths_and_launches():
    import torch

    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    with tvm.target.Target("ascend"):
        kernel = tilelang.compile(_retimed_layout_pair(3, 2), target="ascend", out_idx=[2, 3])
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
