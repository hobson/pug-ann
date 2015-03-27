#!/usr/bin/env python
"""
Uses the unittest module to test this app with `manage.py test`.
"""

# from django.test import TestCase
from unittest import TestCase, main
import doctest
import pip
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
print('Avaialble packages for import!!!!!!!!!')
print(installed_packages_list)
from pug.ann import util
from pug.ann.data import weather


class ANNDocTest(TestCase):

    def test_util(self, module=util):

        failure_count, test_count = doctest.testmod(module, raise_on_error=False, verbose=True)
        msg = "Ran {0} tests in {3} and {1} passed ({2} failed)".format(test_count, test_count-failure_count, failure_count, module.__file__)
        print msg
        if failure_count:
            # print "Ignoring {0} doctest failures...".format(__file__)
            self.fail(msg)
        # return failure_count, test_count

    def test_weather(self):
        self.test_util(weather)

if __name__ == '__main__':
    main()