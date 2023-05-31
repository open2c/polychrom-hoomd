import hoomd
import datetime


class Status():

    def __init__(self, system):
        self.system = system

    @property
    def seconds_remaining(self):
        try:
            return (self.system.final_timestep - self.system.timestep) / self.system.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))
        

class Thermo():

    def __init__(self, system):
        thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        system.operations.computes.append(thermo)
        
        self.thermo = thermo
        
    @property
    def kinetic_temperature(self):
        try:
            return self.thermo.kinetic_temperature
        except hoomd.error.DataAccessError:
            return 0.


def get_logger(system, quantities=[]):
    """Compute logged quantities"""

    status = Status(system)
    thermo = Thermo(system)
    
    logger = hoomd.logging.Logger(categories=['scalar', 'string'])
        
    logger.add(system, quantities=['timestep', 'tps'] + quantities)
    
    logger[('Status', 'etr')] = (status, 'etr', 'string')
    logger[('Thermo', 'kinetic_temperature')] = (thermo, 'kinetic_temperature', 'scalar')

    return logger


def table_formatter(logger, period=5000, **kwargs):
    """Set tabulated logs"""
    
    table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=int(period)),
                              logger=logger, **kwargs)
                             
    return table
