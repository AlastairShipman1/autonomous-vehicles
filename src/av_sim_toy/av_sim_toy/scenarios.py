"""Hand-built scenarios beyond the seeded single-car one, for watching a planner on harder streets.

Each is a fixed ``ScenarioParams`` (no seed) with a one-line description of what it is meant to test. They are
for looking at and for spot checks, not for the tuning/reporting sweeps, which use the sampler.

Pedestrian x is always clear of every parked vehicle (``ScenarioParams`` enforces it).
"""

from __future__ import annotations

from dataclasses import dataclass

from av_sim_toy.scenario import Pedestrian, ScenarioParams, Vehicle


@dataclass(frozen=True)
class NamedScenario:
    name: str
    description: str
    params: ScenarioParams


def _row(xs: list[float], y: float = -2.75, length: float = 4.5) -> tuple[Vehicle, ...]:
    return tuple(Vehicle(x, y, length) for x in xs)


_LIST: list[NamedScenario] = [
    NamedScenario(
        "single_car", "The baseline: one 6 m car, the pedestrian steps out just past its far end.",
        ScenarioParams(initial_speed=11.0, ped_trigger_distance=28.0)),
    NamedScenario(
        "row_of_cars", "Four cars nose to tail (1.5 m gaps); the pedestrian steps out past the last. Hidden for the "
        "whole approach.",
        ScenarioParams(initial_speed=11.0, occluder_x=44.0, occluder_length=4.5, extra_vehicles=_row([50.0, 56.0, 62.0]),
                       ped_x=65.5, ped_trigger_distance=30.0)),
    NamedScenario(
        "gap_between_cars", "Two cars with a 2.5 m gap; the pedestrian steps out of the gap. Visible only when the ego "
        "is straight on to it.",
        ScenarioParams(initial_speed=11.0, occluder_x=46.0, occluder_length=4.5, extra_vehicles=(Vehicle(53.0),),
                       ped_x=49.5, ped_trigger_distance=28.0)),
    NamedScenario(
        "long_truck", "A 12 m truck hides the pedestrian until the ego is almost alongside.",
        ScenarioParams(initial_speed=11.0, occluder_x=50.0, occluder_length=12.0, ped_x=57.0, ped_trigger_distance=28.0)),
    NamedScenario(
        "cars_both_sides", "A parked row on the left adds shadows that matter to nobody; the pedestrian is on the right.",
        ScenarioParams(initial_speed=11.0, occluder_x=50.0, occluder_length=6.0,
                       extra_vehicles=_row([36.0, 43.0, 50.0, 57.0], y=2.75), ped_x=54.0, ped_trigger_distance=28.0)),
    NamedScenario(
        "ped_from_left", "Mirror image: the car and the pedestrian are on the left side of the road.",
        ScenarioParams(initial_speed=11.0, ped_from_left=True, ped_trigger_distance=28.0)),
    NamedScenario(
        "left_cars_ped_right", "Cars on both sides; the pedestrian emerges from the gap in the right-hand pair and crosses through a gap in the left-hand row.",
        ScenarioParams(initial_speed=11.0, occluder_x=46.0, occluder_length=4.5,
                       extra_vehicles=(Vehicle(53.0), *_row([38.0, 44.0, 55.0, 61.0], y=2.75)), ped_x=49.5,
                       ped_trigger_distance=28.0)),
    NamedScenario(
        "dart_out", "A fast pedestrian (2.0 m/s) triggered 26 m out with the ego at 12 m/s: in the lane just as the ego arrives. The hard case.",
        ScenarioParams(initial_speed=12.0, ped_speed=2.0, ped_trigger_distance=26.0)),
    NamedScenario(
        "slow_ped", "A slow pedestrian (0.8 m/s) triggered early, 35 m out. Easy to avoid; tests over-caution.",
        ScenarioParams(initial_speed=11.0, ped_speed=0.8, ped_trigger_distance=35.0)),
    NamedScenario(
        "row_no_pedestrian", "The four-car row with nobody there. Any slowing is over-caution.",
        ScenarioParams(initial_speed=11.0, occluder_x=44.0, occluder_length=4.5, extra_vehicles=_row([50.0, 56.0, 62.0]),
                       ped_present=False)),
    NamedScenario(
        "truck_no_pedestrian", "The 12 m truck with nobody there.",
        ScenarioParams(initial_speed=11.0, occluder_x=50.0, occluder_length=12.0, ped_present=False)),
    NamedScenario(
        "two_peds_one_gap", "Two pedestrians step out of the same 2.5 m gap, the second a moment after the first.",
        ScenarioParams(initial_speed=11.0, occluder_x=46.0, occluder_length=4.5, extra_vehicles=(Vehicle(53.0),),
                       ped_x=49.5, ped_trigger_distance=29.0,
                       extra_pedestrians=(Pedestrian(49.5, speed=1.2, trigger_distance=24.0),))),
    NamedScenario(
        "second_pedestrian_follows", "One pedestrian crosses early and is gone before the ego arrives; a second follows from the "
        "same spot later. The trap for a planner that resumes as soon as the first clears.",
        ScenarioParams(initial_speed=11.0, ped_x=54.0, ped_speed=1.6, ped_trigger_distance=36.0,
                       extra_pedestrians=(Pedestrian(54.0, speed=1.6, trigger_distance=24.0),))),
    NamedScenario(
        "group_of_children", "Three children (1.1 to 1.6 m/s) step out one after another along the far end of the car.",
        ScenarioParams(initial_speed=11.0, ped_x=54.0, ped_speed=1.6, ped_trigger_distance=29.0,
                       extra_pedestrians=(Pedestrian(54.9, speed=1.3, trigger_distance=27.0),
                                          Pedestrian(55.8, speed=1.1, trigger_distance=25.0)))),
    NamedScenario(
        "peds_both_sides", "A pedestrian from behind the right-hand car and another from behind a car on the left, "
        "at different places along the road.",
        ScenarioParams(initial_speed=11.0, occluder_x=52.0, occluder_length=4.5, ped_x=55.0, ped_trigger_distance=29.0,
                       extra_vehicles=(Vehicle(44.0, 2.75),),
                       extra_pedestrians=(Pedestrian(47.4, speed=1.4, trigger_distance=27.0, from_left=True),))),
    NamedScenario(
        "row_two_gaps", "A row of three cars with a 3 m gap between each pair and a pedestrian in each gap.",
        ScenarioParams(initial_speed=11.0, occluder_x=42.0, occluder_length=4.5, extra_vehicles=(Vehicle(49.5), Vehicle(57.0)),
                       ped_x=45.75, ped_trigger_distance=29.0,
                       extra_pedestrians=(Pedestrian(53.25, speed=1.4, trigger_distance=28.0),))),
]

SCENARIOS: dict[str, NamedScenario] = {s.name: s for s in _LIST}
