from typing import List
import torch
import tempfile
import executorchcoreml
from executorch.extension.pybindings import portable_lib as exec_lib

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition.coreml_partitioner import (
    CoreMLPartitioner,
)
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
        _skip_type_promotion=False,  # for future bool(target_precision == torch.float16),
    )

    edge_manager = export_to_edge(
        gm,
        tuple(example_inputs),
        # dynamic_shapes=dynamic_shapes,
        edge_compile_config=edge_config,
    )
    compile_specs = CoreMLBackend.generate_compile_specs(
        model_type=CoreMLBackend.MODEL_TYPE.COMPILED_MODEL
    )
    edge_manager = edge_manager.to_backend(
        CoreMLPartitioner(
            skip_ops_for_coreml_delegation=None, compile_specs=compile_specs
        )
    )
    export_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_constant_segment=False,
            extract_delegate_segments=True,
            passes=[
                QuantFusionPass(),
            ],
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as f:
        export_program.write_to_file(f)
        filename = f.name

    loaded = exec_lib._load_for_executorch(str(filename))

    def run(*args):
        return loaded.forward(args)

    return run


# NOTE: CoreML doesn't like scalars (Rank-0 tensors), which sum() and
# possibly the mul by -1 produces, so I had to stop using this
# example.
@torch.compile(backend=executorch_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


@torch.compile(backend=executorch_compiler)
def tiny_example(a, b):
    return a * a * b


for _ in range(100):
    print(tiny_example(torch.randn(10), torch.randn(10)))
