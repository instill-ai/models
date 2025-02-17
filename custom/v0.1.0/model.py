from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_custom_input,
    construct_custom_output,
)


@instill_deployment
class Custom:
    def __init__(self):
        print("==============yeah init==============")

    async def __call__(self, request):
        custom_inputs = await parse_custom_input(request=request)

        outputs = []
        for inp in custom_inputs:

            outputs.append(inp)

        return construct_custom_output(
            request=request,
            outputs=outputs,
        )


entrypoint = InstillDeployable(Custom).get_deployment_handle()
