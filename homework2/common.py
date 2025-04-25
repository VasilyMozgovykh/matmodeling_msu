import numpy as np
from enum import Enum


class PhysicsEnum(Enum):
    GravityConstant = 6.6743e-11
    AstronomicalUnit = 1.49597870700e11
    EarthMass = 5.9722e24
    SunMass = 1.98892e30


def get_solar_system_params(return_names=False):
    names = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
    ]

    # https://ru.wikipedia.org/wiki/Солнечная_система#Сравнительная_таблица_основных_параметров_планет_и_карликовых_планет
    # https://en.wikipedia.org/wiki/Astronomical_unit
    coords = np.array([
        [0., 0., 0.],
        [0.38 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [0.72 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [1. * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [1.52 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [5.20 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [9.52 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [19.22 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [30.06 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
        [39.2 * PhysicsEnum.AstronomicalUnit.value, 0., 0.],
    ])
    
    # https://en.m.wikipedia.org/wiki/Orbital_speed#Planets
    velocities = np.array([
        [0., 0., 0.],
        [0., 47.36e3, 0.],
        [0., 35.02e3, 0.],
        [0., 29.78e3, 0.],
        [0., 24.13e3, 0.],
        [0., 13.07e3, 0.],
        [0., 9.69e3, 0.],
        [0., 6.81e3, 0.],
        [0., 5.43e3, 0.],
        [0., 4.66e3, 0.],
    ])

    # https://ru.wikipedia.org/wiki/Солнечная_система#Сравнительная_таблица_основных_параметров_планет_и_карликовых_планет
    # https://en.wikipedia.org/wiki/Earth_mass
    masses = np.array([
        PhysicsEnum.SunMass.value,
        0.055 * PhysicsEnum.EarthMass.value,
        0.815 * PhysicsEnum.EarthMass.value,
        1. * PhysicsEnum.EarthMass.value,
        0.107 * PhysicsEnum.EarthMass.value,
        318. * PhysicsEnum.EarthMass.value,
        95. * PhysicsEnum.EarthMass.value,
        14.6 * PhysicsEnum.EarthMass.value,
        17.2 * PhysicsEnum.EarthMass.value,
        0.0022 * PhysicsEnum.EarthMass.value,
    ])

    if return_names:
        return coords, velocities, masses, names
    else:
        return coords, velocities, masses
