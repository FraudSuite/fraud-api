from pydantic import BaseModel
from typing import List
from datetime import datetime


class ModelMetricsSchema(BaseModel):
	model_name: str
	accuracy: float
	precision: float
	recall: float
	f1_score: float
	auc_score: float
	is_production: bool
	sample_count: int
	version: str
	training_time: float
	created_at: datetime


	class Config:
		orm_mode = True
		from_attributes = True