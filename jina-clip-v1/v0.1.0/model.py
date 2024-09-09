# pylint: skip-file
import time
from transformers import AutoModel


from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_embedding_to_multimodal_embedding_input,
    construct_task_embedding_output,
)


@instill_deployment
class Jina:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "jina-clip-v1",
            trust_remote_code=True,
        ).cuda()

    async def __call__(self, request):
        inputs = await parse_task_embedding_to_multimodal_embedding_input(
            request=request
        )

        indexes = []
        created = []
        embeddings = []
        for inp in inputs:
            contents = inp.contents

            indexes_per_seq = []
            created_per_seq = []
            embeddings_per_seq = []
            idx = 0
            i_l = 0
            i_r = 0
            while True:
                if (
                    i_r < len(contents)
                    and contents[i_r]["type"] == contents[i_l]["type"]
                ):
                    i_r += 1
                    continue

                seq_type = contents[i_l]["type"]
                input_values = [v[seq_type] for v in contents[i_l:i_r]]
                if seq_type == "text":
                    output_embeddings = self.model.encode_text(input_values)
                elif seq_type == "image":
                    output_embeddings = self.model.encode_image(input_values)

                for embed in output_embeddings:
                    embeddings_per_seq.append(embed)
                    indexes_per_seq.append(idx)
                    created_per_seq.append(int(time.time()))
                    idx += 1

                if not i_r < len(contents):
                    break

                i_l = i_r

            indexes.append(indexes_per_seq)
            created.append(created_per_seq)
            embeddings.append(embeddings_per_seq)

        return construct_task_embedding_output(
            request=request,
            indexes=indexes,
            created_timestamps=created,
            embeddings=embeddings,
        )


entrypoint = InstillDeployable(Jina).get_deployment_handle()
