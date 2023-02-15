import argparse
import configparser
from typing import List

from faim_hcs.hcs.Experiment import Experiment
from prefect import Flow, Parameter, task, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

from scmultiplex.linking.NucleiLinking import link_nuclei


@task()
def load_experiment(exp_csv):
    exp = Experiment()
    exp.load(exp_csv)
    return exp


@task()
def get_organoids(
    exp: Experiment, exluded_plates: List[str], excluded_wells: List[str]
):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    organoids = []
    for organoid in exp:
        if organoid.well.plate.plate_id not in exluded_plates:
            if organoid.well.well_id not in excluded_wells:
                organoids.append(organoid)

    return organoids


@task()
def get_organoids_task(exp: Experiment, exlude_plates: List[str]):
    return get_organoids(exp, exlude_plates)


@task()
def link_nuclei_task(organoid, ovr_channel, segname, rx_name, RX):
    link_nuclei(
        organoid=organoid,
        ovr_channel=ovr_channel,
        segname=segname,
        rx_name=rx_name,
        RX=RX,
    )


with Flow(
    "Nuclei-Linking", executor=LocalDaskExecutor(), run_config=LocalRun()
) as flow:
    rx_name = Parameter("RX_name", default="R1")
    r0_csv = Parameter("R0_csv", default="/path/to/r0/summary.csv")
    rx_csv = Parameter("RX_csv", default="/path/to/r1/summary.csv")
    excluded_plates = Parameter("excluded_plates", default=[])
    excluded_wells = Parameter("excluded_wells", default=[])
    seg_name = Parameter("seg_name", default="NUC_SEG3D_220523")
    ovr_channel = Parameter("ovr_channel", "C01")

    R0 = load_experiment(r0_csv)
    RX = load_experiment(rx_csv)

    r0_organoids = get_organoids(R0, excluded_plates, excluded_wells)

    link_nuclei_task.map(
        r0_organoids,
        unmapped(ovr_channel),
        unmapped(seg_name),
        unmapped(rx_name),
        unmapped(RX),
    )


def conf_to_dict(config):
    return {
        "R0_csv": config["DEFAULT"]["R0_csv"],
        "RX_csv": config["DEFAULT"]["RX_csv"],
        "RX_name": config["DEFAULT"]["RX_name"],
        "excluded_plates": config["DEFAULT"]["excluded_plates"].split(","),
        "excluded_wells": config["DEFAULT"]["excluded_wells"].split(","),
        "seg_name": config["DEFAULT"]["seg_name"],
        "ovr_channel": config["DEFAULT"]["ovr_channel"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = conf_to_dict(config)
    print(kwargs)

    flow.run(parameters=kwargs)


if __name__ == "__main__":
    main()
