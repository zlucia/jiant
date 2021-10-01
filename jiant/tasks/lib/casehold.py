import pandas as pd
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_file_lines

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
		return self._create_examples(path=self.train_path, set_type="train")

	@classmethod
	def _create_examples(cls, path, set_type):
		df = pd.read_csv(path)
		examples = []
		for i, row in enumerate(df.itertuples()):
			examples.append(
				Example(
					guid="%s-%s" % (set_type, i),
					prompt=row.citing_prompt,
					choice_list=[
						row.holding_0,
						row.holding_1,
						row.holding_2,
						row.holding_3,
						row.holding_4
					],
					label=row.label if set_type != test else cls.CHOICE_KEYS[-1]
					)
				)
		return examples