# pylint: skip-file
import time
from transformers import AutoModel


from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_embedding_to_image_embedding_input,
    construct_task_embedding_output,
)


@instill_deployment
class Clip:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "jina-clip-v1",
            trust_remote_code=True,
        )

    async def __call__(self, request):
        image_inputs = await parse_task_embedding_to_image_embedding_input(
            request=request
        )

        indexes = []
        created = []
        embeddings = []
        for inp in image_inputs:
            output_embeddings = self.model.encode_image(inp.images)

            indexes_per_seq = []
            created_per_seq = []
            embeddings_per_seq = []
            for i, embed in enumerate(output_embeddings):
                embeddings_per_seq.append(embed)
                indexes_per_seq.append(i)
                created_per_seq.append(int(time.time()))

            indexes.append(indexes_per_seq)
            created.append(created_per_seq)
            embeddings.append(embeddings_per_seq)

        return construct_task_embedding_output(
            request=request,
            indexes=indexes,
            created_timestamps=created,
            embeddings=embeddings,
        )


entrypoint = InstillDeployable(Clip).get_deployment_handle()
