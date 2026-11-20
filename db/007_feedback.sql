-- One rating per turn: a second rating for the same turn replaces the first.
CREATE UNIQUE INDEX feedback_turn_id_key ON feedback (turn_id);
GRANT UPDATE (rating, comment) ON feedback TO claims_app;
