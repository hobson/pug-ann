from package_info import __license__, __version__, __name__, __doc__, __author__, __authors__

from pkgutil import extend_path
__path__ = extend_path(__path__, __package__)

import ann
import ann.util
__all__ = ['ann', 'ann.util']
