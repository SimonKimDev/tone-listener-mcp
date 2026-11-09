#import <Foundation/Foundation.h>

#include <string_view>

bool validSinglePresetState(std::string_view json) {
    @autoreleasepool {
        NSData* data = [NSData dataWithBytes:json.data() length:json.size()];
        NSError* error = nil;
        id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (error || ![value isKindOfClass:[NSDictionary class]]) return false;
        NSDictionary* state = value;
        NSDictionary* preset = state[@"preset"];
        NSDictionary* mixer = state[@"mixer"];
        if (![preset isKindOfClass:[NSDictionary class]] ||
            ![mixer isKindOfClass:[NSDictionary class]]) return false;
        NSString* presetId = preset[@"id"];
        if (![presetId isKindOfClass:[NSString class]] || presetId.length == 0 ||
            ![presetId isEqual:state[@"presetId"]]) return false;
        NSArray* slots = mixer[@"activePresetIds"];
        if (![slots isKindOfClass:[NSArray class]] || slots.count != 1 ||
            ![slots[0] isEqual:presetId]) return false;
        NSDictionary* slotData = mixer[@"presetData"];
        if (![slotData isKindOfClass:[NSDictionary class]] || slotData.count != 0) return false;
        NSArray* scenes = preset[@"scenes"];
        if (![scenes isKindOfClass:[NSArray class]] || scenes.count != 1 ||
            ![scenes[0] isKindOfClass:[NSDictionary class]]) return false;
        NSDictionary* scene = scenes[0];
        if (![scene[@"id"] isEqual:state[@"activeSceneId"]]) return false;
        NSDictionary* graph = scene[@"graph"];
        if (![graph isKindOfClass:[NSDictionary class]]) return false;
        NSArray* nodes = graph[@"nodes"];
        if (![nodes isKindOfClass:[NSArray class]] || nodes.count < 2 || nodes.count > 16)
            return false;
        for (id node in nodes) {
            if (![node isKindOfClass:[NSDictionary class]]) return false;
        }
        return [nodes[0][@"type"] isEqual:@"input"] &&
               [nodes[nodes.count - 1][@"type"] isEqual:@"output"];
    }
}
