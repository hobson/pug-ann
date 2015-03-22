from pkg_resources import declare_namespace
declare_namespace(__name__)

import weather

__all__ = globals().get('__all__', []) + ['weather']