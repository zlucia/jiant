from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_jsonl

@dataclass
class Example(mc_template.Example):
	@property
	def task(self):
		return CaseholdTask

@dataclass
class TokenizedExample(mc_template.TokenizedExample):
	pass

@dataclass
class DataRow(mc_template.DataRow):
	pass

@dataclass
class Batch(mc_template.Batch):
	pass

class CaseholdTask(mc_template.AbstractMultipleChoiceTask):
	Example = Example
	TokenizedExample = TokenizedExample
	DataRow = DataRow
	Batch = Batch

	CHOICE_KEYS = [0, 1, 2, 3, 4]
	CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
	NUM_CHOICES = len(CHOICE_KEYS)

	def get_train_examples(self):
		return self._create_examples(lines=read_jsonl(self.train_path), set_type="train")

	@classmethod
	def _create_examples(cls, lines, set_type):
		examples = []
		for (i, line) in enumerate(lines):
			examples.append(
				Example(
					guid="%s-%s" % (set_type, i),
					prompt=line["citing_prompt"],
					choice_list=[
						line["holding_0"],
						line["holding_1"],
						line["holding_2"],
						line["holding_3"],
						line["holding_4"],
					],
					label=line["label"] if set_type != "test" else cls.CHOICE_KEYS[-1]
					)
				)
		return examples