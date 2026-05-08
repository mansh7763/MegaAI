import asyncio
from api.worker.celery_app import celery_app
from api.eval.harness import run_eval


@celery_app.task(name="run_eval_task", bind=True, max_retries=1)
def run_eval_task(self, case_ids=None, run_id=None):
    """Run the evaluation harness asynchronously and return the result dict."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(run_eval(case_ids=case_ids, run_id=run_id))
        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
    finally:
        loop.close()
