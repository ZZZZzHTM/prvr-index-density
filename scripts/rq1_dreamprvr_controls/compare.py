def compare(cells):
 rows=[];checks=[]
 for x in cells:
  seed=x['seed'];ds=x['dataset'];same=x['original_seed']==x['Twin_seed']==x['cliponly_seed']==seed==0
  checks.append(dict(dataset=ds,kind='official_same_seed',passed=same,seeds=[x['original_seed'],x['Twin_seed'],x['cliponly_seed']]))
  orig=x['original']['SumR'];clip=x['ClipOnly']['SumR'];twin=x['Twin']['SumR'];den=orig-clip
  rows.append(dict(dataset=ds,seed=0,author_fusion_weights=x['weights'],original_SumR=orig,ClipOnly_SumR=clip,Twin_SumR=twin,
      ClipOnly_minus_original=clip-orig,Twin_minus_original=twin-orig,recovery_ratio=(twin-clip)/den if den else None,
      recovery_ratio_definition='(Twin-ClipOnly)/(original-ClipOnly)',descriptive_only=True))
 means={key:sum(x[key] for x in rows)/len(rows) for key in ['original_SumR','ClipOnly_SumR','Twin_SumR','ClipOnly_minus_original','Twin_minus_original','recovery_ratio'] if all(x[key] is not None for x in rows)}

 return rows,means,checks
