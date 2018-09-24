
DEBUG = True
LOGIN_DISABLED = True
SECRET_KEY = b'Q\x0c^*)K#<`Kx29Hi=.qg/\rxD>hw65]Sje'
SECURITY_PASSWORD_SALT = b'kB]JW#p\r6|CJWgAI;_]KO\r5sU.R~HMHn'
SECURITY_CHANGEABLE = True
SECURITY_SEND_PASSWORD_CHANGE_EMAIL = True
SECURITY_RECOVERABLE = True
SECURITY_LOGIN_USER_TEMPLATE = 'login.html'
SECURITY_CHANGE_PASSWORD_TEMPLATE = 'change_password.html'
SECURITY_FORGOT_PASSWORD_TEMPLATE = 'forgot_password.html'
SECURITY_RESET_PASSWORD_TEMPLATE = 'reset_password.html'
SECURITY_EMAIL_SENDER = 'noreply@mail.com'

MAIL_SERVER = "smtp.mail.com"
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = "anonymous"
MAIL_PASSWORD = "secret"

CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'

BOOKS_DIR = '/home/user/books/'
IMAGE_SUBDIR = ''
LAREX_DIR = '/home/user/larex_books/'
LAREX_GRP = 'tomcat8'
LAREX_URL_SERVER = 'http://localhost:8080/Larex/'
LAREX_URL_CLIENT = 'http://localhost:8080/Larex/'
