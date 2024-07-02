workers = 4
worker_class = 'uvicorn.workers.UvicornWorker'
bind = '0.0.0.0:8000'
loglevel = 'info'
access_logfile = '/var/log/gunicorn/access.log'
error_logfile = '/var/log/gunicorn/error.log'
