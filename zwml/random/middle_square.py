class middle_square:
    
    def __init__(self):
        pass
    
    def middle_square_list(self, seed, count, width=4, seeds=[]):
        if not seeds:
            assert len(str(seed)) == width, "Seed must have a length equal to request width!"
        x = str(seed**2)
        while len(x)<8:
            x = '0'+ x
        
        spread = width//2
        new_seed = x[width-spread:width+spread]
        seeds.append(new_seed)
        if new_seed == '0000':
            return 'Done'

        count -= 1
        if count == 0:
            return seeds

        return self.middle_square_list(int(new_seed), count, seeds=seeds)
    
    def middle_square_gen(self, seed, width=4):
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