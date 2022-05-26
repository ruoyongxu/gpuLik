x1 <- c(0.05111927, -0.21801231,  0.29980505, -0.19082906, -0.08063808,
        0.36750586, -0.49016828,  0.3477343 ,  0.01152525, -0.40497825,
        0.61173646, -0.49822227,  0.10172736,  0.37866269, -0.68369052,
        0.63531287, -0.2401953 , -0.30441064,  0.71004012, -0.7510192 ,
        0.3903168 ,  0.19296733, -0.69299612,  0.83853477, -0.54062752,
        -0.05400729,  0.63541046, -0.89273724,  0.68107658, -0.10291948,
        -0.54141296,  0.91039872, -0.80297084,  0.26826455,  0.41650909,
        -0.89027745,  0.89910847, -0.43265143, -0.26744892,  0.83312171,
        -0.96395605,  0.58716245,  0.1019733 , -0.74159316,  0.99380774,
        -0.72366505,  0.07151358,  0.62011411, -0.98689599,  0.83513569,
        -0.24431511, -0.47464498,  0.94343789, -0.91595482,  0.40783357,
        0.3124015 , -0.8656101 ,  0.96215416, -0.55396143, -0.14152381,
        0.75745152, -0.97160367,  0.67545418, -0.02928703, -0.62469859,
        0.94412981, -0.76626732,  0.19114667,  0.47456379, -0.8815623 ,
        0.82184112, -0.33535666, -0.31547513,  0.78771199, -0.8393169 ,
        0.4537355 ,  0.15680564, -0.66829072,  0.8176664 , -0.53886649,
        -0.0086435 ,  0.53079758, -0.75770411,  0.58417634, -0.11829242,
        -0.38442355,  0.6619113 , -0.58363162,  0.21232322,  0.24010273,
        -0.53382928,  0.53034529, -0.2591455 , -0.11111424,  0.37584245,
        -0.41056471,  0.2337928 ,  0.01630383, -0.17335079,  0.14097948)


x2<- c(-0.13147935,  0.10756687,  0.08727502, -0.31414053,  0.40669092,
       -0.26988783, -0.05324528,  0.39570299, -0.5576443 ,  0.4241375 ,
       -0.04096955, -0.39859073,  0.65356832, -0.5689592 ,  0.16872249,
       0.34681055, -0.7024288 ,  0.69630034, -0.31455211, -0.25272547,
       0.70679049, -0.79866364,  0.46610769,  0.12633065, -0.6690455 ,
       0.87004782, -0.61266103,  0.02280823,  0.59265057, -0.90626022,
       0.74483019, -0.18513282, -0.48242909,  0.90506029, -0.85464623,
       0.35114962,  0.3445344 , -0.86620595,  0.93571955, -0.51167199,
       -0.18624913,  0.79141662, -0.98341316,  0.65813341,  0.01569005,
       -0.68425792,  0.99498031, -0.78291666,  0.15854432,  0.54995306,
       -0.96964433,  0.87966593, -0.32775744, -0.3951288 ,  0.9086098 ,
       -0.94355991,  0.4835485 ,  0.22750686, -0.81500107,  0.97153024,
       -0.61819673, -0.05555463,  0.69373025, -0.96241481,  0.72501839,
       -0.11188791, -0.55130245,  0.91704032, -0.79867967,  0.26598478,
       0.39557196, -0.83823381,  0.83544925, -0.39825849, -0.23547216,
       0.73076952, -0.83337386,  0.50088672,  0.08075678, -0.60126774,
       0.79235427, -0.56688087,  0.05817623,  0.45808078, -0.71407766,
       0.59001571, -0.17021583, -0.3112461 ,  0.60168002, -0.56413711,
       0.24274739,  0.17272486, -0.45863233,  0.48036822, -0.25815199,
       -0.05776346,  0.2836211 , -0.31182397,  0.17043915, -0.0049786 )

x3<- c(0.99,  0.97,  0.95,  0.93,  0.91,  0.89,  0.87,  0.85,  0.83,
       0.81,  0.79,  0.77,  0.75,  0.73,  0.71,  0.69,  0.67,  0.65,
       0.63,  0.61,  0.59,  0.57,  0.55,  0.53,  0.51,  0.49,  0.47,
       0.45,  0.43,  0.41,  0.39,  0.37,  0.35,  0.33,  0.31,  0.29,
       0.27,  0.25,  0.23,  0.21,  0.19,  0.17,  0.15,  0.13,  0.11,
       0.09,  0.07,  0.05,  0.03,  0.01, -0.01, -0.03, -0.05, -0.07,
       -0.09, -0.11, -0.13, -0.15, -0.17, -0.19, -0.21, -0.23, -0.25,
       -0.27, -0.29, -0.31, -0.33, -0.35, -0.37, -0.39, -0.41, -0.43,
       -0.45, -0.47, -0.49, -0.51, -0.53, -0.55, -0.57, -0.59, -0.61,
       -0.63, -0.65, -0.67, -0.69, -0.71, -0.73, -0.75, -0.77, -0.79,
       -0.81, -0.83, -0.85, -0.87, -0.89, -0.91, -0.93, -0.95, -0.97,
       -0.99)


coords3d <- data.frame(cbind(x1,x2,x3))
save(coords3d, file = "coords3d.RData")





