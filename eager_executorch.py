
from typing import List
import torch
from executorch.extension.pybindings import portable_lib as exec_lib


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # print("my_compiler() called with FX graph:")
    # gm.graph.print_tabular()
    # return gm.forward

    edge_manager = export_to_edge(
            gm,
            input,
            dynamic_shapes=dynamic_shapes,
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
