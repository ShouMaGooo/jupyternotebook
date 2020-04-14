# 宇宙天気　短期的に宇宙線量から予報が可能か検証！？


昨今、温暖化が叫ばれているが、『二酸化炭素の増加』と『気候変動』の相関が見られない事が知られている。
事実ベースだと小氷期や温暖期が周期的に訪れることが知られている。

17世紀にイギリスのテムズ川が凍り、日本でも三大飢饉に見舞われ、1780年の冬にはニューヨーク湾が凍結しています。
逆に、そして、吉野正敏さんの資料からも分かる通り、古墳-飛鳥時代、平安時代初期は、現在よりもむしろ高い気候となっている(PDFの3/16ページ)。

『4～10世紀における気候変動と人間活動』   J-STAGE　　吉野正敏
[https://www.jstage.jst.go.jp/article/jgeography/118/6/118_6_1221/_article/-char/ja/]
※図 1 屋久杉の年輪の炭素同位体比から明らかになった歴史時代の気温変動（北川, 1995a, b）と温暖期・寒冷期の名称（安田, 2004）参照。

そして、長期手に見ると、水蒸気量、太陽風、太陽フレア、太陽地場、地球地場強度、宇宙から降り注ぐ放射線の量『宇宙線量』等の方が、むしろ相関が見られる。
そこで、宇宙線量と長期的には相関がある程度見られるが、短期的に天気への相関が見られるかどうか、練習もかねてピンポイントで予報が可能なのか調べる。


その時に以下のサイトでデータを所得し、それをcsvファイルとして保存。

●横浜　2020年4月（日ごとの値）
https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=46&block_no=47670&year=2020&month=4&day=&view=

●南極昭和基地での宇宙線観測データ　　データの解像度『1日』, 『2019-4-1-00:00 ～ 2019-4-12-00:00』
http://polaris.nipr.ac.jp/~cosmicrays/

#作成CSVファイル名：WeatherYokohama.csv
#Pythonファイル：univerceWeatherYokohama.py
#jupyter notebook の利用時の様子をhtmlファイルでも保管： univerceWeatherYokohama.html

『宇宙線量(ミューオン)』と『気温変化』について  相関係数は、  0.392717  と少し相関が見られるが、単体では難しそうだと分かるが、
従来の観測要素だけでなく、それらと組み合わせることにより気象予報に役立てれるかもしれない。
