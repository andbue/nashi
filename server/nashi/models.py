from nashi.database import Base
from flask_security import UserMixin, RoleMixin
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Boolean, DateTime, Column, Integer, Float, \
                       String, ForeignKey, UnicodeText, PickleType, LargeBinary

"""
get sum over pages:
Page.query.filter(book.name=='Testbuch').with_entities(func.sum(Page.no_lines_segm).label("lines_gt")).first()[0]

book = Book.query.filter_by(name='Testbuch').first()
Page.query.filter_by(book_id=book.id, name="3").first().data
"""


class RolesUsers(Base):
    __tablename__ = 'roles_users'
    id = Column(Integer(), primary_key=True)
    user_id = Column('user_id', Integer(), ForeignKey('user.id'))
    role_id = Column('role_id', Integer(), ForeignKey('role.id'))


class Role(Base, RoleMixin):
    __tablename__ = 'role'
    id = Column(Integer(), primary_key=True)
    name = Column(String(80), unique=True)
    description = Column(String(255))


class User(Base, UserMixin):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True)
    username = Column(String(255))
    password = Column(String(255))
    last_login_at = Column(DateTime())
    current_login_at = Column(DateTime())
    last_login_ip = Column(String(100))
    current_login_ip = Column(String(100))
    login_count = Column(Integer)
    active = Column(Boolean())
    confirmed_at = Column(DateTime())
    roles = relationship('Role', secondary='roles_users',
                         backref=backref('users', lazy='dynamic'))


class Book(Base):
    __tablename__ = 'books'
    id = Column(Integer, primary_key=True)
    name = Column(String(80), unique=True)
    ocrpid = Column(String(80), unique=True)
    no_pages_total = Column(Integer, default=0)
    access = Column(Integer, ForeignKey('user.id'))
    pages = relationship("Page", backref="book")
    models = relationship("OCRModel", backref="book")


class Page(Base):
    __tablename__ = 'pages'
    id = Column(Integer, primary_key=True)
    name = Column(String(80))
    data = Column(UnicodeText(length=2**31))
    no_lines_segm = Column(Integer, default=0)
    no_lines_gt = Column(Integer, default=0)
    no_lines_ocr = Column(Integer, default=0)
    book_id = Column(Integer, ForeignKey('books.id'))


class EditorSettings(Base):
    __tablename__ = 'editorsettings'
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True)
    settings = Column(UnicodeText(length=2**31))


class OCRModel(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    data = Column(LargeBinary)
    trainlines = Column(PickleType)
    testlines = Column(PickleType)
    quality = Column(Float, default=0)
    book_id = Column(Integer, ForeignKey('books.id'))
