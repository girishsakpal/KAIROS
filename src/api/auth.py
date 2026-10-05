"""
Auth placeholder.

The contract requires a JWT on every endpoint with roles ops / exec / admin
(see docs/Kairos_API_Contracts.md, "Cross-cutting notes"). Per the project plan,
JWT is Phase 7 work (Flask-JWT-Extended). Until then `require_role` is a
deliberate NO-OP that marks exactly where enforcement will be added, so the
routes don't need to change later. THE API IS CURRENTLY UNAUTHENTICATED.
"""
from functools import wraps


def require_role(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # TODO (Phase 7): verify_jwt_in_request(); check get_jwt()["role"] in roles
            return fn(*args, **kwargs)
        wrapper.required_roles = roles   # introspectable, handy for the later auth tests
        return wrapper
    return decorator
