for batch in {0..100}; do python parallel_slopes.py extract --name=acute_inflammation --config=inflammation --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=adult --config=income --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=arcene --config=arcene --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=arhythmia --config=has_arhythmia --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=australian_credit --config=australian_credit --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=balance_scale --config=is_balanced --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=bank --config=subscription --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=blood --config=blood --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=breast --config=cancer --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=car --config=car_binary --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=contraceptive --config=contraceptive --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=compas --config=two-years-recidividity --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=covertype --config=covertype_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=dexter --config=dexter --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=electricity --config=electricity --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=fertility --config=fertility --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=german --config=loan --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=gisette --config=gisette --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=glass --config=vehicles --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=heart_failure --config=death --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=heloc --config=risk --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=higgs --config=higgs --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=hill --config=hill --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=hypo --config=has_hypo --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=ipums --config=ipums --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=lrs --config=lrs_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=magic --config=magic --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=madelon --config=madelon --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=house16 --config=house16 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=ionosphere --config=ionosphere --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=magic --config=magic --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=musk --config=musk --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=nbfi --config=default --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=ozone --config=8hr --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=page_blocks --config=page_blocks_binary --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=phoneme --config=phoneme --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=pima --config=pima --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=pol --config=pol --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=pums --config=pums --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=planning --config=planning --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=post_operative --config=post_operative_binary --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=seeds --config=seeds_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=seeds --config=seeds_1 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=seeds --config=seeds_2 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=segment --config=brickface --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=shuttle --config=shuttle_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=sonar --config=sonar --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=spambase --config=spambase --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=spect --config=spect --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=speeddating --config=dating --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=steel_plates --config=steel_plates_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=student_performance --config=math --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=sydt --config=sydt --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=toxicity --config=toxicity --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=twonorm --config=twonorm --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=vertebral_column --config=abnormal --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=wall_following --config=wall_following_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=wine_origin --config=wine_origin_0 --batch=$batch --batch_size=200000 & done
echo "Done."
for batch in {0..100}; do python parallel_slopes.py extract --name=wine --config=wine --batch=$batch --batch_size=200000 & done
echo "Done."