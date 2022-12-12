"""LitBank: Annotated dataset of 100 works of fiction to support tasks in natural language processing and the
computational humanities. """
import pathlib

import datasets
import bratools.anntoconll as brat2conll
from datasets import DatasetInfo

_HOMEPAGE = "https://github.com/dbamman/litbank"

_DESCRIPTION = """\
LitBank is an annotated dataset of 100 works of English-language fiction to support tasks in natural language \
processing and the computational humanities"""

_CITATION = """\
@inproceedings{bamman-etal-2019-annotated,
    title = "An annotated dataset of literary entities",
    author = "Bamman, David  and
      Popat, Sejal  and
      Shen, Sheng",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1220",
    doi = "10.18653/v1/N19-1220",
    pages = "2138--2144",
    abstract = "We present a new dataset comprised of 210,532 tokens evenly drawn from 100 different English-language literary texts annotated for ACE entity categories (person, location, geo-political entity, facility, organization, and vehicle). These categories include non-named entities (such as {``}the boy{''}, {``}the kitchen{''}) and nested structure (such as [[the cook]{'}s sister]). In contrast to existing datasets built primarily on news (focused on geo-political entities and organizations), literary texts offer strikingly different distributions of entity categories, with much stronger emphasis on people and description of settings. We present empirical results demonstrating the performance of nested entity recognition models in this domain; training natively on in-domain literary data yields an improvement of over 20 absolute points in F-score (from 45.7 to 68.3), and mitigates a disparate impact in performance for male and female entities present in models trained on news data.",
}"""

_DOWNLOAD_URL = "https://github.com/dbamman/litbank/archive/refs/heads/master.zip"


class HFLitBankDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coref",
            version=datasets.Version("1.0.0"),
            description="Coreference annotations",
        ),
        datasets.BuilderConfig(
            name="entities",
            version=datasets.Version("1.0.0"),
            description="Named entity annotations",
        ),
        datasets.BuilderConfig(
            name="events",
            version=datasets.Version("1.0.0"),
            description="Event annotations",
        ),
    ]

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("string")),
                    "ner_spans": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int32"))
                    ),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": pathlib.Path(dl_dir) / "litbank-master"}
            ),
        ]

    def _generate_examples(self, filepath):
        data_dir = filepath / self.config.name / "brat"

        for file in data_dir.glob("*.txt"):
            id = file.name.split("_")[0]
            brat2conll.main(["brat2conll", str(file)])
            conll_file = file.with_suffix(".conll")

            with (open(conll_file, "r") as f, open(file, "r") as text):
                tokens = []
                ner_tags = []
                ner_spans = []
                for line in f.readlines():
                    line = line.strip("\n")

                    if line.startswith("#"):
                        continue
                    elif line == "":
                        continue
                    else:
                        line = line.split("\t")
                        if len(line) != 4:
                            continue
                        ner_tags.append(line[0])
                        ner_spans.append([int(line[1]), int(line[2])])
                        tokens.append(line[-1])

                yield id, {
                    "id": id,
                    "text": text.read(),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "ner_spans": ner_spans,
                }
