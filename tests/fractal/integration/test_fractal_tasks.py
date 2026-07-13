import os
from pathlib import Path

import anndata as ad
import numpy as np
from fractal_tasks_core.channels import ChannelInputModel
from numpy.testing import assert_almost_equal

from scmultiplex.fractal.calculate_linking_consensus import calculate_linking_consensus
from scmultiplex.fractal.calculate_object_linking import calculate_object_linking
from scmultiplex.fractal.calculate_platymatch_registration import (
    calculate_platymatch_registration,
)
from scmultiplex.fractal.init_select_all_knowing_reference import (
    init_select_all_knowing_reference,
)
from scmultiplex.fractal.init_select_multiplexing_pairs import (
    init_select_multiplexing_pairs,
)
from scmultiplex.fractal.init_select_reference_knowing_all import (
    init_select_reference_knowing_all,
)
from scmultiplex.fractal.relabel_by_linking_consensus import (
    relabel_by_linking_consensus,
)
from scmultiplex.fractal.scmultiplex_feature_measurements import (
    scmultiplex_feature_measurements,
)
from scmultiplex.fractal.scmultiplex_mesh_measurements import (
    scmultiplex_mesh_measurements,
)
from scmultiplex.fractal.spherical_harmonics_from_labelimage import (
    spherical_harmonics_from_labelimage,
)
from scmultiplex.fractal.surface_mesh_multiscale import surface_mesh_multiscale

name_3d = "220605_151046.zarr"
name_mip = "220605_151046_mip.zarr"

test_calculate_object_linking_expected_output = np.array(
    [[1.0, 1.0, 0.9727956], [2.0, 2.0, 0.8249731], [3.0, 3.0, 0.9644809]]
)

test_calculate_linking_consensus_expected_output = np.array(
    [[1.0, 1.0, 0.0, 1.0], [2.0, 2.0, 1.0, 2.0], [3.0, 3.0, 2.0, 3.0]]
)

test_relabel_by_linking_consensus_output_dict = {
    "0": np.array(
        [
            [13.216666, 21.016666, 0.0, 22.1, 20.583334, 0.6],
            [13.866667, 91.433334, 0.0, 18.85, 15.816667, 0.6],
            [39.0, 103.566666, 0.0, 23.183332, 28.166666, 0.6],
        ]
    ),
    "1": np.array(
        [
            [49.616665, 11.05, 0.0, 22.533333, 20.583334, 0.6],
            [51.783333, 81.9, 0.0, 18.85, 15.816667, 0.6],
            [75.4, 93.816666, 0.0, 23.4, 27.95, 0.6],
        ]
    ),
}

test_calculate_platymatch_registration_output = np.array(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [8.0, 7.0],
        [9.0, 8.0],
        [10.0, 9.0],
        [11.0, 10.0],
        [13.0, 12.0],
        [14.0, 13.0],
        [15.0, 14.0],
        [16.0, 15.0],
        [17.0, 16.0],
        [19.0, 17.0],
        [18.0, 18.0],
        [20.0, 19.0],
    ]
)

test_sphr_harmonics_from_labelimg_expected_output = np.array(
    [10.798882, 9.207903, 13.282207]
)

test_sphr_harmonics_from_mesh_expected_output = np.array(
    [10.046525, 8.232602, 12.735255]
)

test_scmultiplex_mesh_measurements_expected_output = np.array(
    [
        4.2192085e03,
        1.2844763e03,
        1.0172350e00,
        4.5982581e-01,
        9.9415630e-01,
        5.8436790e-03,
        7.8099235e-03,
        1.1332338e00,
        1.0085807e00,
    ]
)

test_masking_feature_measurements_expected_output = np.array(
    [
        [
            15.652262687683105,
            14.886972427368164,
            9.97631549835205,
            3.6025583744049072,
            722.81298828125,
            0.0,
            102.0,
            95.0,
            1627.8079833984375,
            865.0828247070312,
            11.134629249572754,
            0.4440406858921051,
            0.8355419635772705,
            16.99447250366211,
            8.712851524353027,
            0.5126873850822449,
            1.5262718200683594,
            50.0,
            0.0,
            498.6819152832031,
            1852.4942626953125,
            5427.0,
            430.0,
            1362.0,
            1767.0,
            2218.0,
            2732.0,
            3135.0,
            3979.169921875,
            684.8593139648438,
            0.9565579891204834,
            1.4719209671020508,
            15.519001960754395,
            15.023478507995605,
            -0.13326115906238556,
            0.136506587266922,
            9.867411613464355,
            -0.10890321433544159,
        ],
        [
            11.59760570526123,
            4.0896172523498535,
            12.938312530517578,
            4.672168254852295,
            737.31884765625,
            0.0,
            102.0,
            95.0,
            1427.82470703125,
            794.77880859375,
            11.208621978759766,
            0.5163931250572205,
            0.9277031421661377,
            15.167740821838379,
            7.363586902618408,
            0.4854768216609955,
            1.353220820426941,
            50.0,
            0.0,
            484.7750244140625,
            1838.12939453125,
            5502.0,
            385.0,
            1304.0,
            1758.0,
            2231.0,
            2823.0,
            3229.0,
            4061.719970703125,
            733.6224975585938,
            0.8457334637641907,
            0.9606005549430847,
            11.367385864257812,
            3.9895753860473633,
            -0.2302192747592926,
            -0.10004214942455292,
            12.752942085266113,
            -0.1853705644607544,
        ],
        [
            4.036924839019775,
            10.246027946472168,
            12.874675750732422,
            4.64918851852417,
            633.29931640625,
            0.0,
            102.0,
            95.0,
            1204.6883544921875,
            667.8035278320312,
            10.65459156036377,
            0.5256955623626709,
            0.9483318328857422,
            16.64432716369629,
            7.206226348876953,
            0.4329538643360138,
            1.5621742010116577,
            50.0,
            0.0,
            430.7443542480469,
            2103.224365234375,
            5908.0,
            425.0,
            1519.0,
            2040.5,
            2580.0,
            3209.0,
            3609.85009765625,
            4348.0,
            816.13623046875,
            0.5746719241142273,
            0.3113979697227478,
            4.238743782043457,
            10.345292091369629,
            0.20181907713413239,
            0.09926401823759079,
            12.608623504638672,
            -0.26605162024497986,
        ],
        [
            11.045337677001953,
            13.808625221252441,
            18.251331329345703,
            6.590758800506592,
            730.4461669921875,
            0.0,
            102.0,
            95.0,
            1389.292724609375,
            793.0606689453125,
            11.173687934875488,
            0.5257683992385864,
            0.921047031879425,
            14.948514938354492,
            9.596089363098145,
            0.6419426202774048,
            1.3378318548202515,
            50.0,
            0.0,
            469.1253356933594,
            1685.0767822265625,
            5418.0,
            356.0,
            1178.0,
            1617.0,
            2108.0,
            2598.0,
            2910.0,
            3744.360107421875,
            703.0014038085938,
            0.7393988370895386,
            0.9634340405464172,
            11.125523567199707,
            14.001022338867188,
            0.08018530905246735,
            0.19239723682403564,
            17.820568084716797,
            -0.4307650327682495,
        ],
        [
            18.65643882751465,
            17.09329605102539,
            27.4647159576416,
            9.917814254760742,
            46.221500396728516,
            0.0,
            102.0,
            95.0,
            145.0019989013672,
            74.24732971191406,
            4.452614784240723,
            0.3187645673751831,
            0.6225341558456421,
            7.399082183837891,
            3.379702568054199,
            0.45677322149276733,
            1.661738634109497,
            50.0,
            1.0,
            114.99405670166016,
            150.32723999023438,
            229.0,
            102.0,
            139.0,
            149.0,
            159.0,
            171.0,
            179.0,
            195.1999969482422,
            15.778097152709961,
            0.5582362413406372,
            0.8886738419532776,
            18.637325286865234,
            17.072919845581055,
            -0.01911480724811554,
            -0.020376767963171005,
            27.42418098449707,
            -0.04053538665175438,
        ],
        [
            0.9028627276420593,
            11.22231388092041,
            28.072940826416016,
            10.137451171875,
            11.970833778381348,
            1.0,
            102.0,
            95.0,
            35.884334564208984,
            16.759166717529297,
            2.838192939758301,
            0.3335949778556824,
            0.7142857313156128,
            5.246560096740723,
            1.843226432800293,
            0.35132095217704773,
            1.8485565185546875,
            50.0,
            1.0,
            36.320472717285156,
            157.2447052001953,
            214.0,
            112.0,
            147.0,
            156.0,
            167.0,
            177.60000610351562,
            184.8000030517578,
            198.8000030517578,
            16.00446128845215,
            0.36109739542007446,
            0.4396171569824219,
            0.909692645072937,
            11.231372833251953,
            0.006829903461039066,
            0.009059380739927292,
            28.050945281982422,
            -0.021996228024363518,
        ],
        [
            15.14530086517334,
            2.611504316329956,
            28.6558780670166,
            10.347955703735352,
            22.27983283996582,
            0.0,
            102.0,
            95.0,
            73.9375,
            31.293167114257812,
            3.491170883178711,
            0.30133333802223206,
            0.7119712233543396,
            6.573602199554443,
            2.3324413299560547,
            0.3548193871974945,
            1.8829219341278076,
            50.0,
            1.0,
            56.848690032958984,
            146.8912811279297,
            193.0,
            109.0,
            138.0,
            146.0,
            156.0,
            164.0,
            169.0,
            181.10000610351562,
            13.40076732635498,
            0.2523198425769806,
            0.06812165677547455,
            15.13913345336914,
            2.6241185665130615,
            -0.0061676097102463245,
            0.012614196166396141,
            28.65299415588379,
            -0.0028848571237176657,
        ],
        [
            5.596244812011719,
            8.481104850769043,
            8.728958129882812,
            3.1521236896514893,
            515.0275268554688,
            0.0,
            87.0,
            73.0,
            1166.0999755859375,
            579.8389892578125,
            9.945133209228516,
            0.4416666626930237,
            0.8882250189781189,
            13.677024841308594,
            7.737155914306641,
            0.5657045841217041,
            1.3752480745315552,
            50.0,
            0.0,
            389.4295349121094,
            1897.5634765625,
            5050.0,
            466.0,
            1373.0,
            1843.0,
            2310.0,
            2735.60009765625,
            3078.0,
            3946.0,
            673.544189453125,
            0.7481650710105896,
            0.8827877044677734,
            5.717199325561523,
            8.142595291137695,
            0.12095479667186737,
            -0.33850952982902527,
            8.854461669921875,
            0.1255035698413849,
        ],
        [
            13.559160232543945,
            10.615728378295898,
            10.71242618560791,
            3.8683760166168213,
            566.0091552734375,
            1.0,
            87.0,
            73.0,
            1264.1199951171875,
            619.9765014648438,
            10.263014793395996,
            0.4477495551109314,
            0.9129526019096375,
            13.544413566589355,
            6.883767604827881,
            0.5082366466522217,
            1.319730520248413,
            50.0,
            0.0,
            416.80291748046875,
            1515.000732421875,
            4617.0,
            294.0,
            1052.5,
            1435.0,
            1892.0,
            2362.0,
            2634.0,
            3215.06005859375,
            616.7533569335938,
            0.7353883981704712,
            0.6186003088951111,
            13.761065483093262,
            10.093170166015625,
            0.20190471410751343,
            -0.5225582122802734,
            10.572026252746582,
            -0.14039990305900574,
        ],
        [
            12.761711120605469,
            5.051576137542725,
            14.712128639221191,
            5.312713146209717,
            462.6093444824219,
            0.0,
            87.0,
            73.0,
            970.0599975585938,
            528.0123291015625,
            9.595595359802246,
            0.4768873453140259,
            0.8761335611343384,
            13.648948669433594,
            7.531845569610596,
            0.5518260598182678,
            1.4224181175231934,
            50.0,
            0.0,
            360.7010498046875,
            2338.69970703125,
            6482.0,
            528.0,
            1679.0,
            2257.0,
            2850.0,
            3486.0,
            4028.85009765625,
            5264.5400390625,
            929.1395874023438,
            0.8401646614074707,
            1.2200846672058105,
            12.794219970703125,
            5.14639949798584,
            0.03250819817185402,
            0.09482357650995255,
            14.36865520477295,
            -0.3434736728668213,
        ],
        [
            4.911465644836426,
            5.949558734893799,
            15.780560493469238,
            5.698535919189453,
            533.7864990234375,
            0.0,
            87.0,
            73.0,
            1225.70068359375,
            599.781005859375,
            10.064440727233887,
            0.435494989156723,
            0.889968991279602,
            14.388405799865723,
            7.714735507965088,
            0.5361772179603577,
            1.4296278953552246,
            50.0,
            0.0,
            402.7207946777344,
            2458.37060546875,
            7763.0,
            400.0,
            1618.5,
            2280.0,
            3078.5,
            3950.0,
            4685.0,
            6353.0,
            1182.9849853515625,
            1.047947645187378,
            1.519343376159668,
            5.2965240478515625,
            5.5910162925720215,
            0.3850584030151367,
            -0.35854217410087585,
            15.055862426757812,
            -0.7246983051300049,
        ],
        [
            14.648833274841309,
            14.258832931518555,
            27.756000518798828,
            10.02299976348877,
            5.633333206176758,
            0.0,
            87.0,
            73.0,
            11.829999923706055,
            6.450166702270508,
            2.2076096534729004,
            0.4761904776096344,
            0.8733624219894409,
            4.472599029541016,
            1.103938341140747,
            0.2468225359916687,
            2.025991916656494,
            50.0,
            1.0,
            20.865005493164062,
            156.1999969482422,
            208.0,
            117.0,
            145.75,
            156.0,
            165.0,
            177.0,
            183.0500030517578,
            195.02999877929688,
            15.367851257324219,
            0.2877749800682068,
            0.24946080148220062,
            14.646520614624023,
            14.256951332092285,
            -0.002312313299626112,
            -0.0018823089776560664,
            27.745370864868164,
            -0.010628681629896164,
        ],
        [
            6.468323707580566,
            11.02782154083252,
            8.881861686706543,
            3.20733904838562,
            611.6954956054688,
            0.0,
            107.0,
            130.0,
            1251.614013671875,
            659.6069946289062,
            10.53203296661377,
            0.48872536420822144,
            0.9273635745048523,
            14.332930564880371,
            8.801876068115234,
            0.6141016483306885,
            1.360889196395874,
            50.0,
            0.0,
            411.1451721191406,
            1952.6077880859375,
            4483.0,
            447.0,
            1441.0,
            1990.0,
            2409.0,
            2733.0,
            2958.0,
            3520.0,
            635.3492431640625,
            0.180029034614563,
            -0.3212438225746155,
            6.072749137878418,
            11.029096603393555,
            -0.3955746293067932,
            0.0012751187896355987,
            9.250955581665039,
            0.36909380555152893,
        ],
        [
            18.052274703979492,
            6.680866718292236,
            9.67788028717041,
            3.4947900772094727,
            569.1920166015625,
            0.0,
            107.0,
            130.0,
            1164.072021484375,
            616.0894775390625,
            10.28221607208252,
            0.48896631598472595,
            0.9238787293434143,
            13.84778118133545,
            7.669449806213379,
            0.5538396239280701,
            1.346770167350769,
            50.0,
            0.0,
            400.2511901855469,
            1414.1343994140625,
            3469.0,
            366.0,
            1072.75,
            1402.0,
            1682.0,
            1991.0,
            2218.64990234375,
            2724.860107421875,
            457.6045227050781,
            0.5709601044654846,
            0.5346907377243042,
            18.040830612182617,
            6.478199005126953,
            -0.011444833129644394,
            -0.20266765356063843,
            9.610346794128418,
            -0.06753366440534592,
        ],
        [
            17.292715072631836,
            22.516361236572266,
            10.485176086425781,
            3.786313772201538,
            694.308349609375,
            0.0,
            107.0,
            130.0,
            1350.31005859375,
            752.52880859375,
            10.986294746398926,
            0.5141844153404236,
            0.9226335287094116,
            13.472054481506348,
            8.399224281311035,
            0.6234553456306458,
            1.2262600660324097,
            50.0,
            0.0,
            454.2653503417969,
            1906.35888671875,
            4524.0,
            439.0,
            1491.0,
            1871.0,
            2236.0,
            2656.0,
            2983.0,
            3691.510009765625,
            597.9732666015625,
            0.6649439334869385,
            0.8860111832618713,
            17.30130386352539,
            22.554683685302734,
            0.008588314987719059,
            0.0383237861096859,
            10.414021492004395,
            -0.07115528732538223,
        ],
        [
            5.082039833068848,
            20.134084701538086,
            11.78293514251709,
            4.25494909286499,
            605.1044921875,
            0.0,
            107.0,
            130.0,
            1115.5689697265625,
            637.8341674804688,
            10.49406909942627,
            0.5424178242683411,
            0.9486862421035767,
            13.779141426086426,
            7.637562274932861,
            0.5542843341827393,
            1.313040852546692,
            50.0,
            0.0,
            406.9359436035156,
            2008.6104736328125,
            5428.0,
            509.0,
            1536.0,
            1950.0,
            2340.0,
            2852.800048828125,
            3342.89990234375,
            4371.18017578125,
            714.781982421875,
            1.0256491899490356,
            2.0023841857910156,
            5.274370193481445,
            19.925371170043945,
            0.19233019649982452,
            -0.2087140828371048,
            11.656513214111328,
            -0.12642253935337067,
        ],
        [
            11.054340362548828,
            3.7519891262054443,
            13.107474327087402,
            4.733254909515381,
            459.736328125,
            0.0,
            107.0,
            130.0,
            808.4959716796875,
            488.3818359375,
            9.575689315795898,
            0.5686315298080444,
            0.9413461089134216,
            15.005845069885254,
            6.902097225189209,
            0.45996060967445374,
            1.5670771598815918,
            50.0,
            0.0,
            346.8653869628906,
            1745.437255859375,
            4488.0,
            434.0,
            1387.0,
            1735.0,
            2048.0,
            2411.0,
            2649.0,
            3351.159912109375,
            540.787841796875,
            0.6145649552345276,
            1.3802677392959595,
            11.114232063293457,
            3.7621052265167236,
            0.059891071170568466,
            0.01011599786579609,
            12.745145797729492,
            -0.36232832074165344,
        ],
        [
            4.794719696044922,
            11.602705001831055,
            19.902055740356445,
            7.186853408813477,
            561.6996459960938,
            0.0,
            107.0,
            130.0,
            1048.9266357421875,
            595.3588256835938,
            10.23690128326416,
            0.5354994535446167,
            0.9434640407562256,
            12.484410285949707,
            8.07288646697998,
            0.6466373801231384,
            1.2195497751235962,
            50.0,
            0.0,
            383.8812561035156,
            1754.423583984375,
            3909.0,
            420.0,
            1329.0,
            1759.0,
            2183.0,
            2533.0,
            2690.0,
            3064.590087890625,
            590.6201171875,
            0.05161740630865097,
            -0.520435094833374,
            4.623908996582031,
            11.412961959838867,
            -0.1708107888698578,
            -0.1897430717945099,
            19.30722999572754,
            -0.5948266386985779,
        ],
        [
            12.812777519226074,
            22.34168243408203,
            19.539873123168945,
            7.056065082550049,
            593.8096923828125,
            0.0,
            107.0,
            130.0,
            993.719970703125,
            627.3843383789062,
            10.428364753723145,
            0.5975623726844788,
            0.9464846849441528,
            12.153287887573242,
            8.529252052307129,
            0.7018061876296997,
            1.1654068231582642,
            50.0,
            0.0,
            393.0124816894531,
            1868.3240966796875,
            4105.0,
            454.0,
            1492.25,
            1911.0,
            2261.0,
            2563.0,
            2744.949951171875,
            3095.3798828125,
            565.1539306640625,
            -0.13063965737819672,
            -0.36707833409309387,
            12.772505760192871,
            22.379011154174805,
            -0.04027212783694267,
            0.03732871636748314,
            19.085004806518555,
            -0.45486825704574585,
        ],
        [
            14.88194751739502,
            12.78798770904541,
            21.79568099975586,
            7.870662689208984,
            746.0223388671875,
            0.0,
            107.0,
            130.0,
            1307.9473876953125,
            790.1031494140625,
            11.25255298614502,
            0.570376455783844,
            0.9442087411880493,
            13.25941276550293,
            9.808418273925781,
            0.7397325038909912,
            1.1783469915390015,
            50.0,
            0.0,
            460.6004333496094,
            1624.0723876953125,
            4948.0,
            339.0,
            1166.0,
            1590.0,
            1992.0,
            2419.0,
            2693.0,
            3383.050048828125,
            619.624755859375,
            0.6635985374450684,
            0.9616453051567078,
            14.84644603729248,
            12.832807540893555,
            -0.03550124540925026,
            0.04481993615627289,
            21.518592834472656,
            -0.27708709239959717,
        ],
    ],
    dtype=np.float64,
)


test_masking_feature_measurements_expected_columns = [
    "x_pos",
    "y_pos",
    "z_pos",
    "z_pos_pix",
    "volume",
    "is_touching_border_xy",
    "imgdim_x",
    "imgdim_y",
    "area_bbox",
    "area_convhull",
    "equivDiam",
    "extent",
    "solidity",
    "axis_major_length",
    "axis_minor_length",
    "minmajAxisRatio",
    "aspectRatio_equivalentDiameter",
    "imgdim_z",
    "is_touching_border_z",
    "surface_area",
    "C01.intensity_mean",
    "C01.intensity_max",
    "C01.intensity_min",
    "C01.percentile25",
    "C01.percentile50",
    "C01.percentile75",
    "C01.percentile90",
    "C01.percentile95",
    "C01.percentile99",
    "C01.stdev",
    "C01.skew",
    "C01.kurtosis",
    "C01.x_pos_weighted",
    "C01.y_pos_weighted",
    "C01.x_massDisp",
    "C01.y_massDisp",
    "C01.z_pos_weighted",
    "C01.z_massDisp",
]


def select_zarr_urls(name, linking_zenodo_zarrs):
    zarr = None
    for z in linking_zenodo_zarrs:
        if z.endswith(name):
            zarr = z
            break
    assert zarr is not None

    # construct zarr url list
    zarr_urls = [f"{zarr}/C/02/0", f"{zarr}/C/02/1"]

    return zarr_urls


def test_calculate_object_linking(linking_zenodo_zarrs, name=name_mip):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_multiplexing_pairs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org"
        roi_table = "well_ROI_table"
        level = 0

        calculate_object_linking(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            roi_table=roi_table,
            level=level,
            iou_cutoff=0.2,
        )
        output_table_path = f"{zarr_url}/tables/{label_name}_match_table"

        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(output, test_calculate_object_linking_expected_output)


def test_calculate_linking_consensus(linking_zenodo_zarrs, name=name_mip):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_reference_knowing_all(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        roi_table = "org_match_table"

        calculate_linking_consensus(
            zarr_url=zarr_url,
            init_args=init_args,
            roi_table=roi_table,
        )

        output_table_path = f"{zarr_url}/tables/{roi_table}_consensus"

        output = ad.read_zarr(output_table_path).to_df().to_numpy()

        assert_almost_equal(output, test_calculate_linking_consensus_expected_output)


def test_relabel_by_linking_consensus(linking_zenodo_zarrs, name=name_mip):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_all_knowing_reference(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org"
        consensus_table = "org_match_table_consensus"

        relabel_by_linking_consensus(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            consensus_table=consensus_table,
        )

        output_table_path = f"{zarr_url}/tables/{label_name}_linked_ROI_table"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output,
            test_relabel_by_linking_consensus_output_dict[Path(zarr_url).name],
            decimal=3,
        )


def test_calculate_platymatch_registration(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_multiplexing_pairs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )

    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name_to_register = "nuc"
        label_name_obj = "org"
        roi_table = "org_ROI_table"
        channel = ChannelInputModel(wavelength_id="A04_C01")

        calculate_platymatch_registration(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name_to_register=label_name_to_register,
            label_name_obj=label_name_obj,
            roi_table=roi_table,
            level=0,
            save_transformation=True,
            mask_by_parent=True,
            calculate_ffd=True,
            seg_channel=channel,
            volume_filter=True,
            volume_filter_threshold=0.10,
        )

        output_table_path_affine = (
            f"{zarr_url}/tables/{label_name_to_register}_match_table_affine"
        )
        output_affine = ad.read_zarr(output_table_path_affine).to_df().to_numpy()
        output_table_path_ffd = (
            f"{zarr_url}/tables/{label_name_to_register}_match_table_ffd"
        )
        output_ffd = ad.read_zarr(output_table_path_ffd).to_df().to_numpy()
        # test that matches are correct; ignore confidence columns
        assert_almost_equal(
            output_affine[:, 0:2],
            test_calculate_platymatch_registration_output,
            decimal=3,
        )
        assert_almost_equal(
            output_ffd[:, 0:2], test_calculate_platymatch_registration_output, decimal=3
        )


def test_surface_mesh_multiscale(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_reference_knowing_all(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        group_by = "org"
        label_name = "nuc"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            group_by=group_by,
            roi_table=roi_table,
            multiscale=True,
            save_mesh=True,
            expandby_factor=0.6,
            sigma_factor=10,
            canny_threshold=0.3,
            mask_contour_by_parent=False,
            filter_children_by_volume=True,
            child_volume_filter_threshold=0.05,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.98,
            smoothing_iterations=2,
        )

        output_mesh_path = f"{zarr_url}/meshes/{group_by}_from_{label_name}"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_surface_mesh_grouped(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_reference_knowing_all(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "nuc"
        group_by = "org"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            group_by=group_by,
            roi_table=roi_table,
            multiscale=False,
            save_mesh=True,
            expandby_factor=1.0,
            sigma_factor=6,
            canny_threshold=0.2,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.97,
            smoothing_iterations=1,
        )

        output_mesh_path = f"{zarr_url}/meshes/{label_name}_grouped"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_surface_mesh_per_object(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_reference_knowing_all(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org"
        roi_table = "org_ROI_table"

        surface_mesh_multiscale(
            zarr_url=zarr_url,
            init_args=init_args,
            label_name=label_name,
            group_by=None,
            roi_table=roi_table,
            multiscale=False,
            save_mesh=True,
            expandby_factor=1.0,
            sigma_factor=6,
            canny_threshold=0.2,
            mask_contour_by_parent=False,
            filter_children_by_volume=False,
            child_volume_filter_threshold=0.05,
            polynomial_degree=30,
            passband=0.01,
            feature_angle=160,
            target_reduction=0.97,
            smoothing_iterations=1,
        )

        output_mesh_path = f"{zarr_url}/meshes/{label_name}"
        # check that 3 mesh files were written
        assert len(os.listdir(output_mesh_path)) == 3


def test_sphr_harmonics_from_labelimage(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_reference_knowing_all(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        label_name = "org_from_nuc"
        roi_table = "org_from_nuc_ROI_table"

        spherical_harmonics_from_labelimage(
            zarr_url=img["zarr_url"],
            init_args=init_args,
            label_name=label_name,
            roi_table=roi_table,
            lmax=2,
            save_mesh=True,
            save_reconstructed_mesh=True,
        )

        # check that 3 mesh files were written
        output_mesh_path = f"{zarr_url}/meshes/{roi_table}_shaics"
        output_mesh_path_reconstructed = (
            f"{zarr_url}/meshes/{roi_table}_shaics_reconstructed"
        )
        assert len(os.listdir(output_mesh_path)) == 3
        assert len(os.listdir(output_mesh_path_reconstructed)) == 3

        # check that first calculated spherical harmonic is correct
        output_table_path = f"{zarr_url}/tables/{label_name}_harmonics"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output[:, 0], test_sphr_harmonics_from_labelimg_expected_output, decimal=5
        )


def test_scmultiplex_mesh_measurements(linking_zenodo_zarrs, name=name_3d):
    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    parallelization_list = init_select_reference_knowing_all(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=0,
    )
    for img in parallelization_list["parallelization_list"]:
        zarr_url = img["zarr_url"]
        init_args = img["init_args"]
        mesh_name = "org_from_nuc"
        roi_table = "org_from_nuc_ROI_table"
        output_table_name = "mesh_features"

        scmultiplex_mesh_measurements(
            zarr_url=img["zarr_url"],
            init_args=init_args,
            mesh_name=mesh_name,
            roi_table=roi_table,
            output_table_name=output_table_name,
            save_hulls=True,
            calculate_curvature=True,
            calculate_harmonics=True,
            lmax=2,
            translate_to_origin=True,
            save_reconstructed_mesh=True,
        )

        # check that 3 mesh files were written for convex hull
        output_mesh_path_chull = f"{zarr_url}/meshes/{mesh_name}_convex_hull"
        assert len(os.listdir(output_mesh_path_chull)) == 3

        # check that all extracted features are correct for first organoid
        output_table_path = f"{zarr_url}/tables/{output_table_name}"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output[0, :], test_scmultiplex_mesh_measurements_expected_output, decimal=4
        )

        # check that 3 mesh files were written for reconstructed harmonics
        output_mesh_path_reconstructed = f"{zarr_url}/meshes/{mesh_name}_reconstructed"
        assert len(os.listdir(output_mesh_path_reconstructed)) == 3

        # check that first calculated spherical harmonic is correct
        output_table_path = f"{zarr_url}/tables/{output_table_name}_harmonics"
        output = ad.read_zarr(output_table_path).to_df().to_numpy()
        assert_almost_equal(
            output[:, 0], test_sphr_harmonics_from_mesh_expected_output, decimal=4
        )


def test_masking_feature_measurements(linking_zenodo_zarrs, name=name_3d):

    zarr_urls = select_zarr_urls(name, linking_zenodo_zarrs)
    channels = {"C01": ChannelInputModel(wavelength_id="A04_C01")}
    output_table_name = "nuc_featx"

    zarr_url = zarr_urls[0]  # test on first round only

    scmultiplex_feature_measurements(
        zarr_url=zarr_url,
        label_name="nuc",
        output_table_name=output_table_name,
        input_channels=channels,
        input_roi_table_name="org_ROI_table",
        measure_morphology=True,
        measure_surface_area=True,
    )

    # Load expected table & compare
    output_table_path = f"{zarr_url}/tables/{output_table_name}"
    output = ad.read_zarr(output_table_path).to_df()

    actual_columns = output.columns.tolist()

    output = output.to_numpy()

    # from pprint import pprint
    # print("test_masking_feature_measurements_expected_output = np.array(")
    # pprint(output.tolist(), width=120)
    # print(", dtype=np.float64)")

    assert actual_columns == test_masking_feature_measurements_expected_columns
    assert_almost_equal(
        output, test_masking_feature_measurements_expected_output, decimal=3
    )

    #
