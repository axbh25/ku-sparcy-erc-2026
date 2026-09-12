# Day 7 — Live red-bin detection and safe approach

Day 7 detects the large red collection bin from RGB, pairs the observation with
raw depth, transforms it through TF, excludes the held-book vicinity, locks the
bin in odometry, and approaches a stand-off enlarged by the held book's forward
extent.  Premature robot/book contact with the collection bin is a failure because placement belongs to Day 8; the bin's normal table-support contact is ignored.
