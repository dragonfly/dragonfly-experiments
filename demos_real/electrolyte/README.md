This is an example of an electrolyte design task used at the Scott Institute for Energy
at CMU. We wish to optimise an electrolyte design for properties such as the
bulk conductivity and viscosity.

The design space (domain) is specified in config.json with the following in consideration:
- We have a library of 7 lithium salts, LiPF6, LiTFSI, LiBF4, LiCoO2, LiMnO2, LiNiO2,
  LiAlO2 (we will refer to the last four as LiXO2), of which we can use a maximum of 4
  in each design. The 'xxx_present' Boolean variables indicating whether or not each salt
  is present. We have used Boolean variables for LiPF6, LiTFSI, and LiBF4, and a Boolean
  array of size 4 for the LiXO2 salts. The max_num_salts constraint ensures that there are
  at most 4 salts.
- The 'xxx_mol' variables indicate the molarity of each salt. The experimental apparatus
  can only add them in increments of 0.05 and hence they are added as discrete_numeric
  variables with different ranges. As they have different ranges we represent salt
  molarities in different variables. If the ranges are the same, we can group them
  together into a multi-dimensional array as we have done in LiXO2_salts_mol. The total
  salt concentration should be less than 3.8 to avoid crash out and this is indicated in
  the 'max_molarity' constraint.
- The solvent can be composed of 4 ingredients: water, ethyl carbonate, ethyl methyl
  carbonate, and dimethyl carbonate. We represent this space via the 3 dimensional
  'solvent_fractions' variable which represent the fractions of the last 3 ingredients.
  Then (1 - sum(solvent_fractions)) is the fraction of water present. The sum of the three
  fractions should be less than 1, as indicated in the 'solvent_fractions_are_at_most_one'
  constraint. 

