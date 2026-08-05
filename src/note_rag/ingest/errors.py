"""Domain errors returned by the ingestion layer."""


class IngestionError(Exception):
    """Base class for expected ingestion failures."""


class UnsupportedDocumentTypeError(IngestionError):
    pass


class DocumentParsingError(IngestionError):
    pass


class UploadValidationError(IngestionError):
    pass
