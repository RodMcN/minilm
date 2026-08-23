from pydantic import BaseModel
import yaml
import os


class ModelConfig(BaseModel):
    d_model: int
    n_head: int
    n_layers: int


class RunConfig(BaseModel):
    max_steps: int
    micro_batch_size: int
    effective_batch_size: int
    sequence_length: int
    lr: float


class Config(BaseModel):
    model: ModelConfig
    run: RunConfig

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]):
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)
