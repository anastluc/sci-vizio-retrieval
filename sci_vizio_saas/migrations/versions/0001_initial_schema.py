"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-09-26 20:27:00.520566
"""
import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Available on every Heroku Postgres plan; needs superuser/owner rights elsewhere.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('stripe_events',
    sa.Column('id', sa.String(length=255), nullable=False),
    sa.Column('type', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('firebase_uid', sa.String(length=128), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=True),
    sa.Column('display_name', sa.String(length=200), nullable=True),
    sa.Column('plan', sa.String(length=20), server_default='free', nullable=False),
    sa.Column('stripe_customer_id', sa.String(length=255), nullable=True),
    sa.Column('stripe_subscription_id', sa.String(length=255), nullable=True),
    sa.Column('subscription_status', sa.String(length=50), nullable=True),
    sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("plan IN ('free', 'premium')", name='users_plan_check'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('firebase_uid'),
    sa.UniqueConstraint('stripe_customer_id')
    )
    op.create_table('collections',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'name', name='collections_user_name_key')
    )
    op.create_index(op.f('ix_collections_user_id'), 'collections', ['user_id'], unique=False)
    op.create_table('documents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('collection_id', sa.UUID(), nullable=True),
    sa.Column('filename', sa.String(length=500), nullable=False),
    sa.Column('title', sa.String(length=1000), nullable=True),
    sa.Column('storage_key', sa.String(length=500), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('page_count', sa.Integer(), nullable=True),
    sa.Column('figure_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('skipped_figure_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('uploading', 'queued', 'processing', 'ready', 'failed')", name='documents_status_check'),
    sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('storage_key')
    )
    op.create_index('documents_user_created_idx', 'documents', ['user_id', 'created_at'], unique=False)
    op.create_index('documents_user_sha256_key', 'documents', ['user_id', 'sha256'], unique=True, postgresql_where=sa.text('sha256 IS NOT NULL'))
    op.create_index(op.f('ix_documents_collection_id'), 'documents', ['collection_id'], unique=False)
    op.create_table('figures',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('page', sa.Integer(), nullable=False),
    sa.Column('image_index', sa.Integer(), nullable=False),
    sa.Column('storage_key', sa.String(length=500), nullable=False),
    sa.Column('content_type', sa.String(length=100), nullable=False),
    sa.Column('width', sa.Integer(), nullable=False),
    sa.Column('height', sa.Integer(), nullable=False),
    sa.Column('figure_type', sa.String(length=50), nullable=False),
    sa.Column('image_type', sa.String(length=200), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('analysis', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
    sa.Column('embedding_model', sa.String(length=200), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'page', 'image_index', name='figures_document_position_key')
    )
    op.create_index('figures_user_type_idx', 'figures', ['user_id', 'figure_type'], unique=False)
    op.create_index(op.f('ix_figures_document_id'), 'figures', ['document_id'], unique=False)
    op.create_index(op.f('ix_figures_user_id'), 'figures', ['user_id'], unique=False)
    op.create_table('jobs',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('kind', sa.String(length=50), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('max_attempts', sa.Integer(), nullable=False),
    sa.Column('run_after', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('locked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_jobs_document_id'), 'jobs', ['document_id'], unique=False)
    op.create_index('jobs_status_run_after_idx', 'jobs', ['status', 'run_after'], unique=False)


def downgrade():
    op.drop_index('jobs_status_run_after_idx', table_name='jobs')
    op.drop_index(op.f('ix_jobs_document_id'), table_name='jobs')
    op.drop_table('jobs')
    op.drop_index(op.f('ix_figures_user_id'), table_name='figures')
    op.drop_index(op.f('ix_figures_document_id'), table_name='figures')
    op.drop_index('figures_user_type_idx', table_name='figures')
    op.drop_table('figures')
    op.drop_index(op.f('ix_documents_collection_id'), table_name='documents')
    op.drop_index('documents_user_sha256_key', table_name='documents', postgresql_where=sa.text('sha256 IS NOT NULL'))
    op.drop_index('documents_user_created_idx', table_name='documents')
    op.drop_table('documents')
    op.drop_index(op.f('ix_collections_user_id'), table_name='collections')
    op.drop_table('collections')
    op.drop_table('users')
    op.drop_table('stripe_events')
