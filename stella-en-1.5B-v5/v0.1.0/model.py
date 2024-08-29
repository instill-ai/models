# pylint: skip-file
import time
from sentence_transformers import SentenceTransformer

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_embedding_to_text_embedding_input,
    construct_task_embedding_output,
)


@instill_deployment
class Stella:
    def __init__(self):
        self.model = SentenceTransformer(
            "stella_en_1.5B_v5",
            trust_remote_code=True,
            local_files_only=True,
        ).cuda()

    async def __call__(self, request):
        chat_inputs = await parse_task_embedding_to_text_embedding_input(
            request=request
        )

        indexes = []
        created = []
        embeddings = []
        for inp in chat_inputs:
            output_embeddings = self.model.encode(inp.contents, prompt_name="s2s_query")

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


entrypoint = InstillDeployable(Stella).get_deployment_handle()
