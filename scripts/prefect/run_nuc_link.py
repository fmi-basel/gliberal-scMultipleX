from typing import List

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from prefect import Flow, Parameter, task


@task()
def load_experiment(exp_csv):
    exp = Experiment()
    exp.load(exp_csv)
    return exp


def get_organoids(exp: Experiment, exlude_plates: List[str]):
    exp.only_iterate_over_wells(False)
    exp.reset_iterator()

    organoids = []
    for organoid in exp:
        if organoid.well.plate.plate_id not in exlude_plates:
            organoids.append(organoid)

    return organoids


@task()
def get_organoids_task(exp: Experiment, exlude_plates: List[str]):
    return get_organoids(exp, exlude_plates)


@task(nout=2)
def load_organoid_measurement(organoid: OrganoidRecord):
    df_ovr = organoid.well.get_measurement("regionprops_ovr_C01")
    df_ovr = df_ovr.set_index("organoid_id")
    df_org = organoid.get_measurement("regionprops_org_C01")
    return df_ovr, df_org


# @task()
# def load_linking_data(organoid: OrganoidRecord, rx_name: str):
#     link_org = organoid.well.get_measurement("linking_ovr_C01_" + rx_name + "toR0")
#     link_org_dict = link_org.set_index("R0_label").T.to_dict("index")["RX_label"]
#
#
# @task()
# def filter_r0_organoids(r0_organoids):
#     # filtered_orgs = []
#     for organoid in r0_organoids:
#         R0_obj = organoid.organoid_id
#         # R0_id = int(R0_obj.rpartition("_")[2])


with Flow("Nuclei-Linking") as flow:
    rx_name = Parameter("RX_name", default="R1")
    r0_csv = Parameter("R0_csv", default="/path/to/r0/summary.csv")
    rx_csv = Parameter("R1_csv", default="/path/to/r1/summary.csv")

    seg_name = Parameter("seg_name", default="NUC_SEG3D_220523")
    ovr_channel = Parameter("ovr_channel", "C01")

    R0 = load_experiment(r0_csv)
    RX = load_experiment(rx_csv)

    r0_organoids = get_organoids(R0, ["day4p5"])

    df_ovr, df_org = load_organoid_measurement.map(r0_organoids)
