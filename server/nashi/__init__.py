from flask import Flask, url_for, Response, jsonify, request
from flask_security import Security, SQLAlchemySessionUserDatastore
from flask_nav import Nav, register_renderer
from flask_mail import Mail
from flask_bootstrap import Bootstrap
from flask_bootstrap.nav import BootstrapRenderer

from nashi.database import db_session
from nashi.models import User, Role

from celery import Celery


class ownRenderer(BootstrapRenderer):
    def visit_Navbar(self, node):
        root = super().visit_Navbar(node)
        root['class'] = 'navbar navbar-inverse navbar-fixed-top'
        root.children[0]['class'] = 'container'
        return root


def make_celery(app):
    celery = Celery(app.name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)
app.config.from_object('nashi.default_settings')
app.config.from_envvar('NASHI_SETTINGS', silent=True)
mail = Mail(app)
Bootstrap(app)
import nashi.views

nav = Nav()
nav.init_app(app)
register_renderer(app, 'navrender', ownRenderer)
import nashi.navigation

user_datastore = SQLAlchemySessionUserDatastore(db_session, User, Role)
security = Security(app, user_datastore)

celery = make_celery(app)
import nashi.tasks


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()
