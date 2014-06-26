#!/usr/bin/env python
from nupic.research.TP import TP
from nupic.research.spatial_pooler import SpatialPooler as SP
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder as ENC

#predicted_state = tp.getPredictedState()
def reconstruct(sp, tp, enc):
   """Compute probability of each column being active on the next timestep
   given the previous activity."""
   sp_output = tp.columnConfidences()
   """This hasn't been implemented yet as far as I can see.
   It should compute the probability of any given input bit
   from the encoders being active given an output SDR."""
   encoder_output = sp.encodedConfidences(sp_output)
   """This takes the bit probabilities and computes the most
   likely input pattern in whatever data type is encoded."""
   reconstructed_input = enc.decode(encoder_output)
   return reconstructed_input

def compute_IPS(next_input, sp, tp, enc):
   #Reconstruct predicted input
   predicted_input = reconstruct(sp, tp, enc)
   #Compute total on bits in input
   total_input_bits = np.sum(next_input)
   #Find the indices of on bits in next and predicted inputs
   next_input_bits = set(np.nonzero(next_input)[0])
   predicted_input_bits = set(np.nonzero(predicted_input)[0])
   #Count bits in next_input predicted in predicted_input, then divide by
   #total bits
   num_matching_bits = len(next_input_bits.union(predicted_input_bits))
   return float(num_matching_bits)/float(total_input_bits)

tp = TP()
sp = SP()
enc = ENC(10)

compute_IPS(30, sp, tp, enc)
