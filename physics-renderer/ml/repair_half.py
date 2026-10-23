"""Bounded wrapper for frozen half-data correction; original source stays intact."""
import time
START,CPU=time.perf_counter(),time.process_time()
import json
from pathlib import Path
import sys
from hii_sim.io import atomic_json,digest
import repair_campaign as campaign

def main():
    settings=campaign.read('ml/repair_half_plan.json');original=campaign.train;fit_plans=[]
    def short_train(name,x,y,vx,vy,plan,out):
        remaining=settings['whole_job_admission_wall_seconds']-(time.perf_counter()-START)-5
        if remaining<5:raise TimeoutError('Half correction: insufficient prespecified whole-job time before fit')
        actual=dict(plan,fit_wall_seconds=min(settings['optimizer_wall_seconds'],remaining));fit_plans.append(actual)
        return original(name,x,y,vx,vy,actual,out)
    campaign.train=short_train
    atomic_json(campaign.OUT/'half_budget_binding.json',dict(protocol=settings,protocol_sha256=digest('ml/repair_half_plan.json'),wrapper_sha256=digest(__file__),frozen_campaign_source_sha256=digest('ml/repair_campaign.py'),created_before_half_fit=True))
    campaign.status('half_preprocessing_correction');sys.argv=[sys.argv[0],'half'];campaign.main()
    atomic_json(campaign.OUT/'half_budget_receipt.json',dict(wall_seconds=time.perf_counter()-START,cpu_seconds=time.process_time()-CPU,actual_fit_plans=fit_plans,protocol=settings,training=campaign.read(campaign.OUT/'half_corrected/full/training.json'),model_promoted=False))
    campaign.status('half_correction_complete');print('Bounded half-data preprocessing correction complete',flush=True)

if __name__=='__main__':main()
