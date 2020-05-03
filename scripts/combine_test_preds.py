
xs_path= "./models/t5-base_xs_mtp5/predictions_test.lst"
s_path= "./models/t5-base_s_mtp5/predictions_test.lst"
m_path= "./models/t5-base_m_mtp5/predictions_test.lst"
l_path= "./models/t5-base_l_mtp5/predictions_test.lst"
xl_path= "./models/t5-base_xl_mtp5/checkpoint-8000/predictions_test.lst"

output_combined_path = 'predictions.lst'

paths = [xs_path, s_path, m_path, l_path, xl_path]
preds = []

for path in paths:
    with open(path,"r") as f:
        l = f.readlines()
        assert(len(l) == 1767)
        l = [x.strip('\n') for x in l]
    preds.append(l)
assert(len(preds)==5)

with open(output_combined_path, "w") as writer:    
    for i in range(1767):
        writer.write(f"{preds[0][i]},{preds[1][i]},{preds[2][i]},{preds[3][i]},{preds[4][i]}\n")
        
print('success')