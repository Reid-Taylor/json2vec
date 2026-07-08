from datetime import datetime

import lightning.pytorch as lit
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import json2vec as j2v

training_records = pl.read_ndjson("../data/arc_problem_sets_training.jsonl")
evaluation_records = pl.read_ndjson("../data/arc_problem_sets_evaluation.jsonl")

@j2v.preprocess
def pair_examples(record: dict) -> j2v.Observation:
    problem_set = record["problem_set"]
    return j2v.Observation({
        "examples": [
            {"grids": [{"pixels": inp["pixels"]}, {"pixels": out["pixels"]}]}
            for inp, out in zip(problem_set["inputs"], problem_set["outputs"])
        ],
    })

def trainer(logger):
    return lit.Trainer(
        callbacks=[
            j2v.RollbackCheckpoint(monitor="loss/validate", mode="min"),
            EarlyStopping(monitor="loss/validate", mode="min", patience=10),
        ],
        min_epochs=100,
        logger=logger,
        precision="bf16-mixed",
    )


model = j2v.Model.from_tree(
    name="problem_set",
    d_model=8,
    n_layers=1,
    n_heads=2,
    batch_size=4,
    embed=True,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
    examples=j2v.Branch(
        length=5,  # up to 5 (input, output) demonstration pairs per problem_set
        grids=j2v.Branch(
            length=2,  # slot 0 = input, slot 1 = output (set by pair_examples)
            # Dynamic masking scoped to outputs only:
            #   window=1 + branch=True + start=False → candidate is the LAST
            #   fixed slot (index 1 = output). Every other slot is ineligible,
            #   so inputs are never masked. rate=0.4 → each output grid is
            #   hidden with probability 0.4 (independently per example pair).
            mask=j2v.Mask(rate=0.4, window=1, branch=True),
            pixels=j2v.Branch(
                length=900,  # 30x30 ARC grid flattened
                cell=j2v.Category(
                    query="[*].examples[*].grids[*].pixels[*]",
                    size=12,
                ),
            ),
        ),
    ),
)


datamodule = j2v.PolarsDataModule(
    model,
    train=training_records,
    validate=evaluation_records,
    test=evaluation_records,
    preprocessor=pair_examples,
    num_workers=0,
    chunk_batch_size=32,
)
logger = WandbLogger(project="fbc", name=datetime.now().strftime("%Y-%m-%d %H:%M"))

model.update(dropout=0.05)

trainer = trainer(logger)

trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)