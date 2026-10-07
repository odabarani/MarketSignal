# Operating the public research beta

## Deployment shape

Use one Gunicorn web worker and one service instance while the queue remains
process-local. The persistent child handles expensive jobs; increasing web-worker
count does not provide a shared queue. The supplied Render blueprint now probes
`/ready`. The blueprint specifies a paid Starter service with a 1 GB persistent disk
and `MARKETSIGNAL_DATA_DIR=/var/data/marketsignal`. Review current hosting costs
before deploying; free services do not provide the required durable storage. No hosting purchase or production deployment has been
performed by this change.

## Probes

- `GET /health`: process liveness, HTTP 200. Does not test storage or providers.
- `GET /ready`: committed SQLite write and scheduler state; HTTP 200 when healthy,
  HTTP 503 when storage is unavailable/locked or the scheduler is stopped/stalled.
  It creates/empties a small probe table and has a one-second SQLite lock timeout.
- Probes do not start research or download data. Readiness does not establish live
  Yahoo access, model quality, disk capacity, or that the local volume is durable.

Responses include a generated `X-Request-ID`. HTTP logs contain request ID, method,
status and duration, without query strings, request bodies or credentials. Error
responses are generic and include the ID; internal stack traces stay in logs.
Job logs record queued, started, finished and failure events with job IDs. The
queue event includes the originating request ID; duplicates share the original
job. Restrict operator access to internal logs.

## Release checks

Run the full test suite, then start Gunicorn using the repository's command.
Run the deployment smoke check against the intended staging base URL:

```bash
python scripts/smoke.py --base-url "$STAGING_BASE_URL"
python scripts/smoke.py --base-url "$STAGING_BASE_URL" --research
```

The first command checks probes and Dash endpoints. The second submits an AAPL
job, polls completion and downloads its retained JSON report. It consumes a queue
slot, provider requests and CPU, so run it intentionally after releases rather
than on every health poll. It uses the regular input defaults. Passing these checks
does not establish data freshness or licensing; inspect the reported market date.

## Monitoring to configure before launch

Connect the hosting logs to an operator-visible monitor. Alert on sustained ready
probe failures, repeated HTTP 5xx, job failures/timeouts, elevated duration and
queue saturation. Monitor process memory and volume space separately. These
signals and logs are implemented locally; external alert destinations, retention
and incident ownership have not been configured.

## Recovery and rollback

Install from `requirements.lock` with hash verification. Retain the exact source,
lockfile and Python runtime used for a release alongside consistent data-store
backups; report hashes identify those inputs but do not archive source files.
Use `python -m src.replay --report report.json --store snapshots.sqlite` to verify
a report offline with matching inputs. A replay does not use Yahoo, initialize
the store or migrate its schema; missing data and mismatched runtimes fail.

1. For a reported failure, match the response request ID to HTTP logs and the
   queued job ID, then inspect its lifecycle and internal error.
2. For storage failure, check volume availability, free space and SQLite locks.
   Do not delete the store to make readiness pass. SQLite snapshots/reports need
   consistent backups using the SQLite backup API or a stopped service; a bare
   copy of an actively written database is not a tested backup.
3. For provider failure, verify required Yahoo destinations and responses from
   the deployed machine. Avoid repeatedly rerunning expensive research during
   an outage. A local ready probe may still pass while Yahoo is unavailable.
4. Job crashes/timeouts restart the child automatically. Web-process restarts lose
   pending jobs; users must rerun them, while retained reports remain if the volume
   survives. Validate backup restoration and restart behavior in staging.
5. Roll back using the host's previous tested application revision and compatible
   storage. Verify ready probes, research and report download afterward. No live
   rollback, hosted deployment or backup restoration has yet been tested.
