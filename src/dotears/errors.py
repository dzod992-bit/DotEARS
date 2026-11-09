class DotEarsError(Exception):
    """An actionable failure suitable for CLI and agent clients."""

    code = "analysis_error"


class DependencyError(DotEarsError):
    code = "missing_dependency"


class MediaError(DotEarsError):
    code = "invalid_media"


class LimitError(DotEarsError):
    code = "limit_exceeded"


class DownloadError(DotEarsError):
    code = "download_failed"


class ProxyConnectionError(DownloadError):
    code = "proxy_connection_failed"


class ConfigurationError(DotEarsError):
    code = "invalid_configuration"
