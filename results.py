PRECISIONS = [
    (1, 0.84340000000000004),
    (2, 0.87609999999999999),
    (3, 0.88019999999999998),
    (4, 0.89770000000000005),
    (5, 0.91049999999999998),
    (6, 0.91890000000000005),
    (7, 0.91869999999999996),
    (8, 0.92200000000000004),
    (9, 0.92169999999999996),
    (10, 0.92300000000000004),
    (15, 0.92849999999999999),
    (20, 0.93410000000000004),
    (30, 0.93540000000000001),
    (50, 0.93130000000000002),
    (70, 0.93020000000000003),
    (100, 0.92100000000000004),
    (150, 0.90739999999999998),
    (200, 0.90100000000000002)
]

LOGS = '''
learning 1 components
training label 0 (5923 samples)
 > converged in 3 iterations in 0:00:00.279583
training label 1 (6742 samples)
 > converged in 3 iterations in 0:00:00.359141
training label 2 (5958 samples)
 > converged in 3 iterations in 0:00:00.375519
training label 3 (6131 samples)
 > converged in 3 iterations in 0:00:00.397705
training label 4 (5842 samples)
 > converged in 3 iterations in 0:00:00.416293
training label 5 (5421 samples)
 > converged in 3 iterations in 0:00:00.420813
training label 6 (5918 samples)
 > converged in 3 iterations in 0:00:00.447617
training label 7 (6265 samples)
 > converged in 3 iterations in 0:00:00.472810
training label 8 (5851 samples)
 > converged in 3 iterations in 0:00:00.452967
training label 9 (5949 samples)
 > converged in 3 iterations in 0:00:00.468124
1 0.8434
learning 2 components
training label 0 (5923 samples)
 > converged in 22 iterations in 0:00:03.443867
training label 1 (6742 samples)
 > converged in 6 iterations in 0:00:01.065651
training label 2 (5958 samples)
 > converged in 8 iterations in 0:00:01.245395
training label 3 (6131 samples)
 > converged in 4 iterations in 0:00:00.642668
training label 4 (5842 samples)
 > converged in 10 iterations in 0:00:01.541051
training label 5 (5421 samples)
 > converged in 7 iterations in 0:00:00.994652
training label 6 (5918 samples)
 > converged in 7 iterations in 0:00:01.083951
training label 7 (6265 samples)
 > converged in 37 iterations in 0:00:06.157154
training label 8 (5851 samples)
 > converged in 3 iterations in 0:00:00.458204
training label 9 (5949 samples)
 > converged in 47 iterations in 0:00:07.345552
2 0.8761
learning 3 components
training label 0 (5923 samples)
 > converged in 3 iterations in 0:00:00.665403
training label 1 (6742 samples)
 > converged in 3 iterations in 0:00:00.757228
training label 2 (5958 samples)
 > converged in 5 iterations in 0:00:01.099957
training label 3 (6131 samples)
 > converged in 10 iterations in 0:00:02.277523
training label 4 (5842 samples)
 > converged in 5 iterations in 0:00:01.098984
training label 5 (5421 samples)
 > converged in 3 iterations in 0:00:00.600325
training label 6 (5918 samples)
 > converged in 85 iterations in 0:00:18.962207
training label 7 (6265 samples)
 > converged in 58 iterations in 0:00:13.767135
training label 8 (5851 samples)
 > converged in 4 iterations in 0:00:00.861311
training label 9 (5949 samples)
 > converged in 122 iterations in 0:00:27.347910
3 0.8802
learning 4 components
training label 0 (5923 samples)
 > converged in 5 iterations in 0:00:01.419626
training label 1 (6742 samples)
 > converged in 14 iterations in 0:00:04.544444
training label 2 (5958 samples)
 > converged in 4 iterations in 0:00:01.124743
training label 3 (6131 samples)
 > converged in 11 iterations in 0:00:03.234964
training label 4 (5842 samples)
 > converged in 7 iterations in 0:00:01.980159
training label 5 (5421 samples)
 > converged in 13 iterations in 0:00:03.361211
training label 6 (5918 samples)
 > converged in 22 iterations in 0:00:06.375393
training label 7 (6265 samples)
 > converged in 18 iterations in 0:00:05.504519
training label 8 (5851 samples)
 > converged in 3 iterations in 0:00:00.834566
training label 9 (5949 samples)
 > converged in 63 iterations in 0:00:18.286959
4 0.8977
learning 5 components
training label 0 (5923 samples)
 > converged in 12 iterations in 0:00:04.276828
training label 1 (6742 samples)
 > converged in 24 iterations in 0:00:09.866685
training label 2 (5958 samples)
 > converged in 10 iterations in 0:00:03.557883
training label 3 (6131 samples)
 > converged in 27 iterations in 0:00:09.816351
training label 4 (5842 samples)
 > converged in 31 iterations in 0:00:10.828978
training label 5 (5421 samples)
 > converged in 47 iterations in 0:00:15.297514
training label 6 (5918 samples)
 > converged in 22 iterations in 0:00:07.862104
training label 7 (6265 samples)
 > converged in 44 iterations in 0:00:16.537031
training label 8 (5851 samples)
 > converged in 4 iterations in 0:00:01.381455
training label 9 (5949 samples)
 > converged in 15 iterations in 0:00:05.305119
5 0.9105
learning 6 components
training label 0 (5923 samples)
 > converged in 6 iterations in 0:00:02.462387
training label 1 (6742 samples)
 > converged in 16 iterations in 0:00:07.591905
training label 2 (5958 samples)
 > converged in 7 iterations in 0:00:02.856501
training label 3 (6131 samples)
 > converged in 122 iterations in 0:00:52.954752
training label 4 (5842 samples)
 > converged in 7 iterations in 0:00:02.861286
training label 5 (5421 samples)
 > converged in 5 iterations in 0:00:01.867635
training label 6 (5918 samples)
 > converged in 7 iterations in 0:00:02.896568
training label 7 (6265 samples)
 > converged in 66 iterations in 0:00:29.949314
training label 8 (5851 samples)
 > converged in 36 iterations in 0:00:14.894701
training label 9 (5949 samples)
 > converged in 67 iterations in 0:00:28.369183
6 0.9189
learning 7 components
training label 0 (5923 samples)
 > converged in 14 iterations in 0:00:06.854412
training label 1 (6742 samples)
 > converged in 4 iterations in 0:00:02.182480
training label 2 (5958 samples)
 > converged in 29 iterations in 0:00:14.246965
training label 3 (6131 samples)
 > converged in 8 iterations in 0:00:03.979864
training label 4 (5842 samples)
 > converged in 7 iterations in 0:00:03.339900
training label 5 (5421 samples)
 > converged in 19 iterations in 0:00:08.414549
training label 6 (5918 samples)
 > converged in 167 iterations in 0:01:22.235817
training label 7 (6265 samples)
 > converged in 21 iterations in 0:00:10.842267
training label 8 (5851 samples)
 > converged in 31 iterations in 0:00:14.785109
training label 9 (5949 samples)
 > converged in 14 iterations in 0:00:06.804930
7 0.9187
learning 8 components
training label 0 (5923 samples)
 > converged in 13 iterations in 0:00:07.166784
training label 1 (6742 samples)
 > converged in 5 iterations in 0:00:03.109270
training label 2 (5958 samples)
 > converged in 31 iterations in 0:00:17.227664
training label 3 (6131 samples)
 > converged in 10 iterations in 0:00:05.588039
training label 4 (5842 samples)
 > converged in 33 iterations in 0:00:18.427358
training label 5 (5421 samples)
 > converged in 27 iterations in 0:00:13.655949
training label 6 (5918 samples)
 > converged in 14 iterations in 0:00:07.658448
training label 7 (6265 samples)
 > converged in 9 iterations in 0:00:05.228617
training label 8 (5851 samples)
 > converged in 13 iterations in 0:00:06.943633
training label 9 (5949 samples)
 > converged in 56 iterations in 0:00:30.844914
8 0.922
learning 9 components
training label 0 (5923 samples)
 > converged in 19 iterations in 0:00:11.684711
training label 1 (6742 samples)
 > converged in 15 iterations in 0:00:10.527954
training label 2 (5958 samples)
 > converged in 25 iterations in 0:00:15.495535
training label 3 (6131 samples)
 > converged in 48 iterations in 0:00:30.952600
training label 4 (5842 samples)
 > converged in 14 iterations in 0:00:08.630606
training label 5 (5421 samples)
 > converged in 14 iterations in 0:00:07.840379
training label 6 (5918 samples)
 > converged in 13 iterations in 0:00:07.986761
training label 7 (6265 samples)
 > converged in 15 iterations in 0:00:09.832274
training label 8 (5851 samples)
 > converged in 6 iterations in 0:00:03.579640
training label 9 (5949 samples)
 > converged in 37 iterations in 0:00:23.022440
9 0.9217
learning 10 components
training label 0 (5923 samples)
 > converged in 11 iterations in 0:00:07.492552
training label 1 (6742 samples)
 > converged in 37 iterations in 0:00:28.894706
training label 2 (5958 samples)
 > converged in 6 iterations in 0:00:04.017177
training label 3 (6131 samples)
 > converged in 12 iterations in 0:00:08.448933
training label 4 (5842 samples)
 > converged in 9 iterations in 0:00:06.065412
training label 5 (5421 samples)
 > converged in 38 iterations in 0:00:24.033502
training label 6 (5918 samples)
 > converged in 4 iterations in 0:00:02.650782
training label 7 (6265 samples)
 > converged in 11 iterations in 0:00:07.969286
training label 8 (5851 samples)
 > converged in 7 iterations in 0:00:04.612840
training label 9 (5949 samples)
 > converged in 10 iterations in 0:00:06.895061
10 0.923
earning 15 components
training label 0 (5923 samples)
 > converged in 37 iterations in 0:00:38.171299
training label 1 (6742 samples)
 > converged in 9 iterations in 0:00:10.382023
training label 2 (5958 samples)
 > converged in 10 iterations in 0:00:10.258782
training label 3 (6131 samples)
 > converged in 13 iterations in 0:00:13.709012
training label 4 (5842 samples)
 > converged in 33 iterations in 0:00:34.164128
training label 5 (5421 samples)
 > converged in 16 iterations in 0:00:14.983736
training label 6 (5918 samples)
 > converged in 7 iterations in 0:00:07.072853
training label 7 (6265 samples)
 > converged in 44 iterations in 0:00:48.506133
training label 8 (5851 samples)
 > converged in 16 iterations in 0:00:15.861782
training label 9 (5949 samples)
 > converged in 3 iterations in 0:00:02.960377
15 0.9285
learning 20 components
training label 0 (5923 samples)
iteration 14 (elapsed 0:00:19.080968)
 > converged in 16 iterations in 0:00:21.846771
training label 1 (6742 samples)
 > converged in 19 iterations in 0:00:29.290035
training label 2 (5958 samples)
 > converged in 13 iterations in 0:00:17.897368
training label 3 (6131 samples)
 > converged in 38 iterations in 0:00:54.283560
training label 4 (5842 samples)
 > converged in 12 iterations in 0:00:16.315234
training label 5 (5421 samples)
 > converged in 7 iterations in 0:00:08.562056
training label 6 (5918 samples)
 > converged in 9 iterations in 0:00:12.013285
training label 7 (6265 samples)
 > converged in 8 iterations in 0:00:11.463267
training label 8 (5851 samples)
 > converged in 48 iterations in 0:01:04.524069
training label 9 (5949 samples)
 > converged in 34 iterations in 0:00:47.162486
20 0.9341
learning 30 components
training label 0 (5923 samples)
 > converged in 35 iterations in 0:01:12.853902
training label 1 (6742 samples)
 > converged in 107 iterations in 0:04:08.350860
training label 2 (5958 samples)
 > converged in 14 iterations in 0:00:28.793483
training label 3 (6131 samples)
 > converged in 19 iterations in 0:00:40.311483
training label 4 (5842 samples)
 > converged in 29 iterations in 0:01:00.743596
training label 5 (5421 samples)
 > converged in 25 iterations in 0:00:47.556130
training label 6 (5918 samples)
 > converged in 15 iterations in 0:00:30.674163
training label 7 (6265 samples)
 > converged in 26 iterations in 0:00:57.490940
training label 8 (5851 samples)
 > converged in 19 iterations in 0:00:38.140509
training label 9 (5949 samples)
 > converged in 33 iterations in 0:01:08.398129
30 0.9354
learning 50 components
training label 0 (5923 samples)
 > converged in 14 iterations in 0:00:48.531836
training label 1 (6742 samples)
 > converged in 50 iterations in 0:03:14.094876
training label 2 (5958 samples)
 > converged in 21 iterations in 0:01:14.800028
training label 3 (6131 samples)
 > converged in 24 iterations in 0:01:29.532684
training label 4 (5842 samples)
 > converged in 6 iterations in 0:00:21.151666
training label 5 (5421 samples)
 > converged in 42 iterations in 0:02:23.757412
training label 6 (5918 samples)
 > converged in 24 iterations in 0:01:32.256708
training label 7 (6265 samples)
 > converged in 39 iterations in 0:02:41.626084
training label 8 (5851 samples)
 > converged in 15 iterations in 0:00:58.416358
training label 9 (5949 samples)
 > converged in 21 iterations in 0:01:33.070000
50 0.9313
learning 70 components
training label 0 (5923 samples)
 > converged in 12 iterations in 0:01:04.941534
training label 1 (6742 samples)
 > converged in 6 iterations in 0:00:35.820462
training label 2 (5958 samples)
 > converged in 12 iterations in 0:01:05.019630
training label 3 (6131 samples)
 > converged in 45 iterations in 0:04:09.419728
training label 4 (5842 samples)
 > converged in 25 iterations in 0:02:14.943062
training label 5 (5421 samples)
 > converged in 16 iterations in 0:01:19.675939
training label 6 (5918 samples)
 > converged in 21 iterations in 0:01:55.935117
training label 7 (6265 samples)
 > converged in 24 iterations in 0:02:16.681590
training label 8 (5851 samples)
 > converged in 22 iterations in 0:01:57.182564
training label 9 (5949 samples)
 > converged in 20 iterations in 0:01:47.122959
70 0.9302
'''
