import logging
from functools import wraps

from flask import jsonify, make_response, request
from substrateinterface import Keypair, KeypairType

logger = logging.getLogger("waitress")
logger.setLevel(logging.DEBUG)


def authenticate_request_with_bittensor(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        message = data.get("message") if data else None
        signature = data.get("signature") if data else None
        public_address = data.get("public_address") if data else None

        if not (message and signature and public_address):
            logger.info(f"Rejected request without auth data")
            return make_response(
                jsonify({"error": "Missing message, signature, or public_address"}), 400
            )

        # For simplicity, we're not checking against a metagraph here.
        # In a real implementation, you'd want to verify the miner is registered on the network.

        signature_bytes = (
            bytes.fromhex(signature) if isinstance(signature, str) else signature
        )
        keypair_public = Keypair(
            ss58_address=public_address, crypto_type=KeypairType.SR25519
        )
        is_valid = keypair_public.verify(message.encode("utf-8"), signature_bytes)

        if is_valid:
            return f(*args, **kwargs)
        else:
            logger.info(
                f"Miner {public_address} refused. Signature Verification Failed"
            )
            return make_response(
                jsonify({"error": "Signature verification failed"}), 403
            )

    return decorated_function
