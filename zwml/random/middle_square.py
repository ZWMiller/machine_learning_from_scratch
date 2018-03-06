class middle_square:
    
    def __init__(self):
        """
        Generates random numbers using a middle square method. 
        Squares the seed, pads the left side of the number with 
        zeroes, then takes the middle values as the next random
        number in the sequence. Note: do not use in production,
        very easy to crack.
        """
        pass
    
    def middle_square_list(self, seed, count, width=4, seeds=[]):
        """
        Creates a list of length "count" of random numbers
        given a seed, by squaring the seed and taking the middle
        digits. If the seed becomes 0000, stops early.
        Works recursively by creating one value at a time and 
        sending that value to the next call as the new seed.
        ---
        KWargs:
        seed: starting value for the RNG
        count: how many numbers to generate
        width: how many digits is the generated number
        seeds: stores the results so far, can be used to force
        a certain number to be in the result.
        """
        if not seeds:
            assert len(str(seed)) == width, "Seed must have a length equal to request width!"
        x = str(seed**2)
        while len(x)<width:
            x = '0'+ x
        
        spread = width//2
        new_seed = x[width-spread:width+spread]
        seeds.append(new_seed)
        if new_seed == ''.join(['0' for _ in range(width)]):
            return 'Done'

        count -= 1
        if count == 0:
            return seeds

        return self.middle_square_list(int(new_seed), count, width=width, seeds=seeds)
    
    def middle_square_gen(self, seed, width=4):
        """
        Generates random numbers given a seed, by squaring the seed 
        and taking the middle digits. Each number
        will have number of digits equal to width. This is a
        generator, so it must be handled as such.
        ---
        KWargs:
        seed: starting value for the RNG
        width: how many digits is the generated number
        """
        assert len(str(seed)) == width, "Seed must have a length equal to request width!"
        new_seed = seed
        while True:
            x = str(int(new_seed)**2)
            while len(x)<2*width:
                x = '0'+ x
            spread = width//2
            new_seed = x[width-spread:width+spread]

            if int(new_seed) == 0:
                new_seed = seed

            yield int(new_seed)