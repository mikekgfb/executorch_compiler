
from typing import List
import torch
from executorch.extension.pybindings import portable_lib as exec_lib

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)

# TODO: change back to executorch.examples.portable.utils
# when executorch installs correctly

from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch_portable_utils import export_to_edge


def executorch_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # print("my_compiler() called with FX graph:")
    # gm.graph.print_tabular()
    # return gm.forward

    # TODO: need to use kv sdpa?
    edge_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_type_promotion=False, # for future bool(target_precision == torch.float16),
    )
    
    edge_manager = export_to_edge(
        gm,
        tuple(example_inputs),
        # dynamic_shapes=dynamic_shapes,
        edge_compile_config=edge_config,
    )
    edge_manager = edge_manager.to_backend(XnnpackDynamicallyQuantizedPartitioner())
    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_constant_segment=True,
            extract_delegate_segments=True,
            passes=[
                QuantFusionPass(),
            ],
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        export_program.write_to_file(f)
        filename = f.name

    return exec_lib._load_for_executorch(str(path)).forward

@torch.compile(backend=executorch_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

for _ in range(100):
    print(toy_example(torch.randn(10), torch.randn(10)))
