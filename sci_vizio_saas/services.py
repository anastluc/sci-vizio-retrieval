"""Long-lived external dependencies, built lazily and replaceable in tests."""

from functools import cached_property

from sci_vizio_saas.settings import Settings


class Services:
    """
    Holds storage, identity, billing, embedding and vision clients.

    Pass any of them to the constructor to override the default (e.g. fakes in
    tests); anything not passed is built from settings on first use.
    """

    def __init__(self, settings: Settings, **overrides):
        self.settings = settings
        for name, value in overrides.items():
            if not isinstance(getattr(type(self), name, None), cached_property):
                raise TypeError(f"Unknown service {name!r}")
            self.__dict__[name] = value

    @cached_property
    def storage(self):
        from sci_vizio_saas.storage import make_storage

        return make_storage(self.settings)

    @cached_property
    def identity(self):
        from sci_vizio_saas.auth import FirebaseIdentityProvider

        return FirebaseIdentityProvider(
            project_id=self.settings.firebase_project_id,
            credentials_json=self.settings.firebase_credentials_json,
            require_verified_email=self.settings.require_verified_email,
        )

    @cached_property
    def billing(self):
        """Stripe gateway, or None when billing isn't configured."""
        if not self.settings.billing_enabled:
            return None
        from sci_vizio_saas.billing import StripeGateway

        return StripeGateway(self.settings.stripe_secret_key, self.settings.stripe_price_id,
                             self.settings.stripe_webhook_secret)

    @cached_property
    def embedder(self):
        from sci_vizio_retrieval.embeddings import FastEmbedEmbedder
        from sci_vizio_saas.models import EMBEDDING_DIMENSION

        embedder = FastEmbedEmbedder(self.settings.embedding_model, cache_dir=self.settings.embedding_cache_dir)
        if embedder.dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding model {self.settings.embedding_model!r} produces {embedder.dimension}-d vectors; "
                f"the schema stores {EMBEDDING_DIMENSION}-d vectors (needs a migration)."
            )
        return embedder

    @cached_property
    def vision(self):
        from openai import OpenAI

        from sci_vizio_retrieval.client import OpenRouterVision

        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        client = OpenAI(base_url=self.settings.openrouter_base_url, api_key=self.settings.openrouter_api_key)
        return OpenRouterVision(model=self.settings.vision_model, api_delay=0, max_retries=3,
                                max_tokens=self.settings.vision_max_tokens, client=client)
