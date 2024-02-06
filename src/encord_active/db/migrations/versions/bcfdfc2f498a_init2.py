"""init2

Revision ID: bcfdfc2f498a
Revises: 1bdba1540905
Create Date: 2023-07-27 11:21:43.336331+01:00

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "bcfdfc2f498a"
down_revision = "1bdba1540905"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        "active_data_project_hash_metric_annotation_quality_index",
        table_name="active_project_analytics_data",
    )
    op.drop_column("active_project_analytics_data", "metric_annotation_quality")
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "active_project_analytics_data",
        sa.Column("metric_annotation_quality", sa.FLOAT(), nullable=True),
    )
    op.create_index(
        "active_data_project_hash_metric_annotation_quality_index",
        "active_project_analytics_data",
        ["project_hash", "metric_annotation_quality"],
        unique=False,
    )
    # ### end Alembic commands ###
