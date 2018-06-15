from nashi import nav
from flask_security.core import current_user
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator


@nav.navigation()
def mynavbar():
    if current_user.is_anonymous:
        usergroup = Subgroup(
                    "User",
                    Text("Not logged in"),
                    View('Forgot password', 'security.forgot_password'),
                )
    else:
        usergroup = Subgroup(
                    "User",
                    Text(current_user.email),
                    View('Change password', 'security.change_password'),
                    Separator,
                    View('Logout', 'security.logout')
                )

    return Navbar(
        View('nashi', 'index'),
        Subgroup(
            'Links',
            Link('Philosophie Uni Würzburg', 'https://www.philosophie.' +
                 'uni-wuerzburg.de/forschung/forschungsstellephilosophie-un/'),
            Link('AL-Corpus', 'http://arabic-latin-corpus.philosophie.' +
                 'uni-wuerzburg.de/'),
            Link('AL-Corpus wiki', 'http://kallimachos.de/al-corpus'),
        ),
        usergroup
    )


@nav.navigation()
def indexnavbar():
    nav = mynavbar()
    nav.items.insert(0, Subgroup(
            'Library',
            View('Refresh list of books', 'index', action="refresh_booklist"),
            ))
    return nav
