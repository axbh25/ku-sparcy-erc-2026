# KU SPARCy — Emirates Robotics Competition 2026

Private development repository for Khalifa University's KU SPARCy team.

## Required evaluation command

```bash
ros2 launch ku_sparcy_erc solution.launch.py shelf_column_number:=2 book_colour:=red
```

Day 1 implements and validates the fixed Phase 1 opening turn, live camera input,
and head positioning. The complete autonomous mission will replace this temporary
opening-only implementation during Days 2–9.
