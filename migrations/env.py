"""Alembic migration environment."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from note_rag.persistence.models import Base
from note_rag.persistence.settings import DatabaseSettings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option(
    "sqlalchemy.url",
    DatabaseSettings.from_env().url.replace("%", "%%"),
)
target_metadata = Base.metadata
managed_table_names = set(target_metadata.tables)


def include_managed_names(
    name: str | None,
    type_: str,
    parent_names,
) -> bool:
    """Avoid reflecting tables that this migration history does not own."""

    if type_ == "table":
        return name in managed_table_names
    return True


def include_managed_objects(
    object_,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to,
) -> bool:
    """Keep Alembic away from tables owned by LangChain or other components."""

    if type_ == "table":
        return name in managed_table_names
    parent_table = getattr(object_, "table", None)
    if parent_table is not None:
        return parent_table.name in managed_table_names
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_name=include_managed_names,
        include_object=include_managed_objects,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_name=include_managed_names,
            include_object=include_managed_objects,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
