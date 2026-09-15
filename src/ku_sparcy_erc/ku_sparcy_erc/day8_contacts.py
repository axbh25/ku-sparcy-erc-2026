"""Phase-dependent contact evidence. IDs are opaque sensor labels, not pose oracles."""
from dataclasses import dataclass, field
from typing import Optional


def names_of(contact):
    return tuple(str(getattr(getattr(contact, key, None), 'name', ''))
                 for key in ('collision1', 'collision2'))


def is_bin(name):
    return 'collection_bin' in name or 'colleciton_bin' in name


def is_table(name):
    return 'table' in name.lower()


def is_robot(name):
    return any(t in name for t in ('tiago_pro::', 'arm_left_', 'arm_right_',
        'gripper_left_', 'gripper_right_', 'torso_', 'head_', '::base_link::',
        'wheel_', 'suspension_')) and not is_bin(name) and not is_table(name)


@dataclass
class ContactLedger:
    arm: str
    held_id: Optional[str] = None
    tip_last: Optional[float] = None
    bin_last: Optional[float] = None
    ignored_support: int = 0
    robot_bin: int = 0
    unexpected: int = 0
    premature: int = 0
    support_samples: list = field(default_factory=list)
    tip_samples: list = field(default_factory=list)
    history: list = field(default_factory=list)
    lock_count: int = 0

    def feed(self, pair, now, phase, allow_support=False):
        a, b = pair
        if not a or not b:
            self.unexpected += 1
            return 'CONTACT_LABEL_MISSING'
        tips = (f'gripper_{self.arm}_fingertip_left_link',
                f'gripper_{self.arm}_fingertip_right_link')
        tip_a, tip_b = any(t in a for t in tips), any(t in b for t in tips)
        bin_a, bin_b = is_bin(a), is_bin(b)
        self.history.append({'time': now, 'phase': phase, 'a': a, 'b': b})
        self.history = self.history[-180:]
        if bin_a or bin_b:
            other = b if bin_a else a
            if is_table(other):
                self.ignored_support += 1
                return None
            if is_robot(other):
                self.robot_bin += 1
                return 'ROBOT_BIN_CONTACT'
            if self.held_id is not None and other == self.held_id:
                if not allow_support:
                    self.premature += 1
                    return 'PREMATURE_HELD_OBJECT_BIN_CONTACT'
                self.bin_last = now
                if not self.support_samples or now > self.support_samples[-1] + 1e-6:
                    self.support_samples.append(now)
                self.support_samples = self.support_samples[-500:]
                return None
            self.unexpected += 1
            return 'UNIDENTIFIED_OBJECT_BIN_CONTACT'
        if tip_a or tip_b:
            other = b if tip_a else a
            if is_robot(other) or is_table(other):
                self.unexpected += 1
                return 'FINGERTIP_NONBOOK_CONTACT'
            identity = other
            if self.held_id is None:
                self.held_id = identity
            if identity != self.held_id:
                self.unexpected += 1
                return 'HELD_OBJECT_ID_CHANGED'
            self.lock_count += 1
            self.tip_last = now
            if not self.tip_samples or now > self.tip_samples[-1] + 1e-6:
                self.tip_samples.append(now)
            self.tip_samples = self.tip_samples[-500:]
            return None
        if is_robot(a) or is_robot(b):
            # Wheel-ground support is not a collision penalty. All other
            # unexpected arm/base/head/environment contacts fail closed.
            if any('wheel_' in n for n in pair) and any('ground' in n for n in pair):
                return None
            self.unexpected += 1
            return 'UNINTENDED_ROBOT_CONTACT'
        return None

    def fresh_tip(self, now, age=0.8):
        return self.tip_last is not None and 0 <= now-self.tip_last <= age

    def support_window(self, now, duration, age=0.45, minimum=3, after=-1.0):
        times = [t for t in self.support_samples if t >= after and t >= now-duration-age]
        return (len(times) >= minimum and now-times[-1] <= age
                and times[-1]-times[0] >= duration
                and max((b-a for a,b in zip(times,times[1:])), default=0) <= age)

    def as_dict(self):
        return dict(held_object_contact_id=self.held_id,
                    last_selected_fingertip_time=self.tip_last,
                    last_held_object_bin_time=self.bin_last,
                    ignored_bin_table_messages=self.ignored_support,
                    robot_bin_contacts=self.robot_bin,
                    unexpected_contacts=self.unexpected,
                    premature_book_bin_contacts=self.premature,
                    support_timestamps=self.support_samples,
                    fingertip_timestamps=self.tip_samples,
                    contact_history=self.history)
